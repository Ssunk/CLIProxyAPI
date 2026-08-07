// Package vision implements automatic image-to-text fallback for models that do
// not support image input. When a blacklisted model receives a request
// containing images, each image part is replaced by the text description
// produced by a separate vision-capable model before the request is forwarded
// upstream.
//
// The package operates on the client's original (source-format) JSON payload and
// is format-aware: OpenAI chat, Claude, Gemini, and Codex/Responses image shapes
// are all recognized. All failures are fail-open: the original body is returned
// unchanged so the proxy never makes a request worse than the status quo.
package vision

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	// defaultVisionPrompt is used when no prompt is configured.
	defaultVisionPrompt = "Describe the image content in detail, including any visible text, objects, and layout. Output only the description."
	// defaultVisionMaxTokens caps the vision model description length.
	defaultVisionMaxTokens = 512
	// maxVisionResponseBytes bounds how much of the vision response is read.
	maxVisionResponseBytes = 1 << 20 // 1 MiB
	// maxErrorLogBytes bounds how much of an error body is logged.
	maxErrorLogBytes = 4 << 10 // 4 KiB
)

// imageRef describes one image part found in a source-format payload.
type imageRef struct {
	// path is the sjson path to the part to replace, e.g. "messages.0.content.1".
	path string
	// dataURI is the image passed to the vision model: a "data:<mime>;base64,..."
	// URI or an http(s) URL.
	dataURI string
	// hash is the sha256 hex of dataURI, used to dedup identical images.
	hash string
}

// Apply rewrites body so every image part is replaced by the text description
// produced by the vision model. It returns the body to forward: the original
// body when the feature is disabled, the model is not blacklisted, the source
// format has no images, or a vision request fails (fail-open). It never returns
// nil and never aborts the request.
func Apply(ctx context.Context, body []byte, sourceFormat, requestedModel, routedModel string, cfg *config.VisionFallbackConfig, client *http.Client) []byte {
	if len(body) == 0 || cfg == nil || !cfg.Enabled {
		return body
	}
	if !blacklisted(requestedModel, cfg.Models) && !blacklisted(routedModel, cfg.Models) {
		return body
	}
	refs := findImages(body, sourceFormat)
	if len(refs) == 0 || client == nil {
		return body
	}
	if ctx == nil {
		ctx = context.Background()
	}

	out := body
	descriptions := make(map[string]string, len(refs))
	for _, ref := range refs {
		desc, ok := descriptions[ref.hash]
		if !ok {
			var err error
			desc, err = describeImage(ctx, client, cfg, ref.dataURI)
			if err != nil {
				log.WithContext(ctx).WithFields(log.Fields{
					"source_format": sourceFormat,
					"model":         routedModel,
					"image_hash":    ref.hash[:12],
					"error":         err,
				}).Warnf("vision fallback: describing image failed; forwarding original request")
				return body
			}
			descriptions[ref.hash] = desc
		}
		updated, err := sjson.SetRawBytes(out, ref.path, buildTextPart(sourceFormat, desc))
		if err != nil {
			log.WithContext(ctx).WithFields(log.Fields{
				"source_format": sourceFormat,
				"model":         routedModel,
				"error":         err,
			}).Warnf("vision fallback: failed to replace image part; forwarding original request")
			return body
		}
		out = updated
	}
	return out
}

// blacklisted reports whether model matches any entry in list, compared
// case-insensitively after trimming whitespace.
func blacklisted(model string, list []string) bool {
	needle := strings.ToLower(strings.TrimSpace(model))
	if needle == "" {
		return false
	}
	for _, entry := range list {
		if strings.ToLower(strings.TrimSpace(entry)) == needle {
			return true
		}
	}
	return false
}

// findImages locates every image part in a source-format payload. Formats not
// listed (e.g. antigravity) and formats without chat images are treated as
// having no images.
func findImages(body []byte, sourceFormat string) []imageRef {
	switch sourceFormat {
	case "openai", "claude":
		return findImagesInMessages(body, sourceFormat)
	case "gemini":
		return findImagesInGemini(body)
	case "codex", "openai-response":
		return findImagesInInput(body)
	default:
		return nil
	}
}

// findImagesInMessages scans messages[].content[] for image parts (OpenAI chat
// image_url and Claude image/url parts).
func findImagesInMessages(body []byte, sourceFormat string) []imageRef {
	var refs []imageRef
	gjson.GetBytes(body, "messages").ForEach(func(mi, m gjson.Result) bool {
		content := m.Get("content")
		if content.IsArray() {
			content.ForEach(func(pi, part gjson.Result) bool {
				if dataURI, ok := imageURLFromChatPart(part, sourceFormat); ok {
					refs = append(refs, newRef(fmt.Sprintf("messages.%s.content.%s", mi.String(), pi.String()), dataURI))
				}
				return true
			})
		}
		return true
	})
	return refs
}

// findImagesInGemini scans contents[].parts[] for inlineData/inline_data parts.
func findImagesInGemini(body []byte) []imageRef {
	var refs []imageRef
	gjson.GetBytes(body, "contents").ForEach(func(ci, c gjson.Result) bool {
		parts := c.Get("parts")
		if parts.IsArray() {
			parts.ForEach(func(pi, part gjson.Result) bool {
				for _, key := range []string{"inlineData", "inline_data"} {
					inline := part.Get(key)
					if !inline.Exists() {
						continue
					}
					mime := inline.Get("mimeType").String()
					if mime == "" {
						mime = inline.Get("mime_type").String()
					}
					data := inline.Get("data").String()
					if mime == "" || data == "" {
						return true
					}
					refs = append(refs, newRef(fmt.Sprintf("contents.%s.parts.%s", ci.String(), pi.String()), "data:"+mime+";base64,"+data))
					return true
				}
				return true
			})
		}
		return true
	})
	return refs
}

// findImagesInInput scans codex/openai-response input[] for top-level input_image
// items and input_image parts nested inside message items' content arrays.
// Items backed only by a server-side file_id cannot be resolved and are skipped.
func findImagesInInput(body []byte) []imageRef {
	var refs []imageRef
	input := gjson.GetBytes(body, "input")
	if !input.IsArray() {
		return nil
	}
	input.ForEach(func(ii, item gjson.Result) bool {
		switch item.Get("type").String() {
		case "input_image":
			if dataURI, ok := imageURLFromInputImage(item); ok {
				refs = append(refs, newRef("input."+ii.String(), dataURI))
			}
		case "message":
			content := item.Get("content")
			if content.IsArray() {
				content.ForEach(func(pi, part gjson.Result) bool {
					if part.Get("type").String() == "input_image" {
						if dataURI, ok := imageURLFromInputImage(part); ok {
							refs = append(refs, newRef(fmt.Sprintf("input.%s.content.%s", ii.String(), pi.String()), dataURI))
						}
					}
					return true
				})
			}
		}
		return true
	})
	return refs
}

// imageURLFromChatPart extracts the image reference from an OpenAI or Claude
// content part. It accepts OpenAI image_url parts, Claude image parts (base64 or
// url sources), and Claude bare url parts.
func imageURLFromChatPart(part gjson.Result, sourceFormat string) (string, bool) {
	switch part.Get("type").String() {
	case "image_url":
		if url := part.Get("image_url.url").String(); url != "" {
			return url, true
		}
	case "image":
		src := part.Get("source")
		switch src.Get("type").String() {
		case "base64":
			mime := src.Get("media_type").String()
			data := src.Get("data").String()
			if mime != "" && data != "" {
				return "data:" + mime + ";base64," + data, true
			}
		case "url":
			if url := src.Get("url").String(); url != "" {
				return url, true
			}
		}
	case "url":
		// Claude content part may use a top-level "url" for an image reference.
		if url := part.Get("url").String(); url != "" {
			return url, true
		}
	}
	return "", false
}

// imageURLFromInputImage extracts the image reference from an input_image item.
// image_url may be a plain string or an object with a url field.
func imageURLFromInputImage(item gjson.Result) (string, bool) {
	if url := item.Get("image_url.url").String(); url != "" {
		return url, true
	}
	if imageURL := item.Get("image_url"); imageURL.Exists() && imageURL.Type == gjson.String && imageURL.String() != "" {
		return imageURL.String(), true
	}
	return "", false
}

// newRef builds an imageRef, computing the dedup hash from the data URI.
func newRef(path, dataURI string) imageRef {
	sum := sha256.Sum256([]byte(dataURI))
	return imageRef{path: path, dataURI: dataURI, hash: hex.EncodeToString(sum[:])}
}

// buildTextPart builds a replacement part carrying the description. Replacement
// is 1-for-1 so array indices never shift. The part shape matches the source
// format so downstream translators still recognize it.
func buildTextPart(sourceFormat, desc string) []byte {
	text := "[Image: " + desc + "]"
	switch sourceFormat {
	case "codex", "openai-response":
		b, _ := json.Marshal(map[string]any{"type": "input_text", "text": text})
		return b
	case "gemini":
		b, _ := json.Marshal(map[string]any{"text": text})
		return b
	default:
		b, _ := json.Marshal(map[string]any{"type": "text", "text": text})
		return b
	}
}

// buildVisionPayload builds an OpenAI-compatible chat completions request that
// asks the vision model to describe a single image.
func buildVisionPayload(dataURI string, cfg *config.VisionFallbackConfig) []byte {
	prompt := strings.TrimSpace(cfg.Prompt)
	if prompt == "" {
		prompt = defaultVisionPrompt
	}
	maxTokens := cfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultVisionMaxTokens
	}
	payload := map[string]any{
		"model": cfg.Model,
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": prompt},
					map[string]any{"type": "image_url", "image_url": map[string]any{"url": dataURI}},
				},
			},
		},
		"max_tokens": maxTokens,
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return nil
	}
	return b
}

// describeImage sends one image to the vision model and returns its description.
func describeImage(ctx context.Context, client *http.Client, cfg *config.VisionFallbackConfig, dataURI string) (string, error) {
	endpoint := strings.TrimSuffix(strings.TrimSpace(cfg.BaseURL), "/") + "/chat/completions"
	payload := buildVisionPayload(dataURI, cfg)
	if len(payload) == 0 {
		return "", errors.New("failed to build vision request payload")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey := strings.TrimSpace(cfg.APIKey); apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer func() {
		_ = resp.Body.Close()
	}()
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, maxVisionResponseBytes))
	if err != nil {
		return "", err
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return "", fmt.Errorf("vision endpoint returned %d: %s", resp.StatusCode, truncateLog(respBody))
	}
	return parseVisionResponse(respBody)
}

// parseVisionResponse extracts the description text from an OpenAI-compatible
// chat completions response. The content may be a plain string or an array of
// content parts.
func parseVisionResponse(respBody []byte) (string, error) {
	content := gjson.GetBytes(respBody, "choices.0.message.content")
	var text string
	switch {
	case content.Type == gjson.String:
		text = content.String()
	case content.IsArray():
		var parts []string
		content.ForEach(func(_, p gjson.Result) bool {
			if p.Get("type").String() == "text" {
				parts = append(parts, p.Get("text").String())
			}
			return true
		})
		text = strings.Join(parts, "\n")
	}
	text = strings.TrimSpace(text)
	if text == "" {
		return "", errors.New("vision response contained no text content")
	}
	return text, nil
}

// truncateLog bounds a loggable string to maxErrorLogBytes.
func truncateLog(b []byte) string {
	if len(b) > maxErrorLogBytes {
		b = b[:maxErrorLogBytes]
	}
	return string(b)
}
