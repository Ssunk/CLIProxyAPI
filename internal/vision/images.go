package vision

import (
	"bytes"
	"fmt"
	"strconv"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/constant"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// resolveText maps an image URL (https or data URL; "" when unresolvable) to
// the replacement marker text.
type resolveText func(imageURL string) string

type imageTarget struct {
	path string
	url  string
}

// hasImageHint is a cheap pre-check before any JSON traversal. False positives
// are fine; false negatives are not.
func hasImageHint(format string, body []byte) bool {
	switch format {
	case constant.Claude:
		return bytes.Contains(body, []byte(`"image"`))
	case constant.OpenAI:
		return bytes.Contains(body, []byte(`"image_url"`))
	case constant.OpenaiResponse:
		return bytes.Contains(body, []byte(`"input_image"`)) || bytes.Contains(body, []byte(`"image_url"`))
	case constant.Gemini:
		return bytes.Contains(body, []byte("inlineData")) || bytes.Contains(body, []byte("inline_data")) ||
			bytes.Contains(body, []byte("fileData")) || bytes.Contains(body, []byte("file_data"))
	}
	return false
}

// textPartJSON builds a replacement content part with proper JSON escaping.
// partType is "text" (claude/openai) or "input_text" (responses).
func textPartJSON(partType, text string) []byte {
	part := []byte(`{"type":"","text":""}`)
	part, _ = sjson.SetBytes(part, "type", partType)
	part, _ = sjson.SetBytes(part, "text", text)
	return part
}

// geminiTextPartJSON builds a bare Gemini text part.
func geminiTextPartJSON(text string) []byte {
	part := []byte(`{"text":""}`)
	part, _ = sjson.SetBytes(part, "text", text)
	return part
}

func applyTargets(body []byte, targets []imageTarget, partType string, resolve resolveText) ([]byte, int) {
	out := body
	for _, target := range targets {
		var replacement []byte
		if partType == "" {
			replacement = geminiTextPartJSON(resolve(target.url))
		} else {
			replacement = textPartJSON(partType, resolve(target.url))
		}
		if updated, errSet := sjson.SetRawBytes(out, target.path, replacement); errSet == nil {
			out = updated
		}
	}
	return out, len(targets)
}

// claudeImageURL extracts a URL for a Claude image block: base64 sources
// become data URLs, url sources pass through. "" means unresolvable.
func claudeImageURL(part gjson.Result) string {
	source := part.Get("source")
	switch source.Get("type").String() {
	case "base64":
		mediaType := source.Get("media_type").String()
		data := source.Get("data").String()
		if mediaType == "" || data == "" {
			return ""
		}
		return "data:" + mediaType + ";base64," + data
	case "url":
		return source.Get("url").String()
	}
	return ""
}

// replaceClaudeImages replaces image blocks in Claude Messages requests,
// including images nested inside tool_result blocks.
func replaceClaudeImages(body []byte, resolve resolveText) ([]byte, int) {
	messages := gjson.GetBytes(body, "messages")
	if !messages.IsArray() {
		return body, 0
	}
	targets := make([]imageTarget, 0, 4)
	messageIndex := 0
	messages.ForEach(func(_, message gjson.Result) bool {
		content := message.Get("content")
		if content.IsArray() {
			partIndex := 0
			content.ForEach(func(_, part gjson.Result) bool {
				switch part.Get("type").String() {
				case "image":
					targets = append(targets, imageTarget{
						path: fmt.Sprintf("messages.%d.content.%d", messageIndex, partIndex),
						url:  claudeImageURL(part),
					})
				case "tool_result":
					nested := part.Get("content")
					if nested.IsArray() {
						nestedIndex := 0
						nested.ForEach(func(_, nestedPart gjson.Result) bool {
							if nestedPart.Get("type").String() == "image" {
								targets = append(targets, imageTarget{
									path: fmt.Sprintf("messages.%d.content.%d.content.%d", messageIndex, partIndex, nestedIndex),
									url:  claudeImageURL(nestedPart),
								})
							}
							nestedIndex++
							return true
						})
					}
				}
				partIndex++
				return true
			})
		}
		messageIndex++
		return true
	})
	return applyTargets(body, targets, "text", resolve)
}

// replaceOpenAIImages replaces image_url parts in OpenAI chat-completions
// requests for all roles, including tool messages.
func replaceOpenAIImages(body []byte, resolve resolveText) ([]byte, int) {
	messages := gjson.GetBytes(body, "messages")
	if !messages.IsArray() {
		return body, 0
	}
	targets := make([]imageTarget, 0, 4)
	messageIndex := 0
	messages.ForEach(func(_, message gjson.Result) bool {
		content := message.Get("content")
		if content.IsArray() {
			partIndex := 0
			content.ForEach(func(_, part gjson.Result) bool {
				if part.Get("type").String() == "image_url" {
					targets = append(targets, imageTarget{
						path: fmt.Sprintf("messages.%d.content.%d", messageIndex, partIndex),
						url:  part.Get("image_url.url").String(),
					})
				}
				partIndex++
				return true
			})
		}
		messageIndex++
		return true
	})
	return applyTargets(body, targets, "text", resolve)
}

// responsesOutputImageURL reports whether the part is an image part in either
// the Responses ("input_image" with a string image_url) or chat-completions
// ("image_url" with an object image_url) shape used by function_call_output
// payloads, returning the image URL.
func responsesOutputImageURL(part gjson.Result) (string, bool) {
	switch part.Get("type").String() {
	case "input_image":
		return part.Get("image_url").String(), true
	case "image_url":
		return part.Get("image_url.url").String(), true
	}
	return "", false
}

// replaceResponsesOutputParts replaces image parts inside a standalone JSON
// array (the decoded form of a string-encoded function_call_output payload).
func replaceResponsesOutputParts(arr []byte, resolve resolveText) ([]byte, int) {
	parsed := gjson.ParseBytes(arr)
	if !parsed.IsArray() {
		return arr, 0
	}
	targets := make([]imageTarget, 0, 2)
	partIndex := 0
	parsed.ForEach(func(_, part gjson.Result) bool {
		if imageURL, ok := responsesOutputImageURL(part); ok {
			targets = append(targets, imageTarget{path: strconv.Itoa(partIndex), url: imageURL})
		}
		partIndex++
		return true
	})
	return applyTargets(arr, targets, "input_text", resolve)
}

// replaceResponsesImages replaces input_image parts in OpenAI Responses
// requests, including images inside function_call_output payloads in their
// structured-array and JSON-encoded-string forms.
func replaceResponsesImages(body []byte, resolve resolveText) ([]byte, int) {
	input := gjson.GetBytes(body, "input")
	if !input.IsArray() {
		return body, 0
	}
	targets := make([]imageTarget, 0, 4)
	type stringOutputTarget struct {
		path  string
		inner string
	}
	stringOutputs := make([]stringOutputTarget, 0, 2)
	itemIndex := 0
	input.ForEach(func(_, item gjson.Result) bool {
		itemType := item.Get("type").String()
		if itemType == "message" || itemType == "" {
			content := item.Get("content")
			if content.IsArray() {
				partIndex := 0
				content.ForEach(func(_, part gjson.Result) bool {
					if part.Get("type").String() == "input_image" {
						targets = append(targets, imageTarget{
							path: fmt.Sprintf("input.%d.content.%d", itemIndex, partIndex),
							url:  part.Get("image_url").String(),
						})
					}
					partIndex++
					return true
				})
			}
		}
		if itemType == "function_call_output" {
			output := item.Get("output")
			switch {
			case output.IsArray():
				partIndex := 0
				output.ForEach(func(_, part gjson.Result) bool {
					if imageURL, ok := responsesOutputImageURL(part); ok {
						targets = append(targets, imageTarget{
							path: fmt.Sprintf("input.%d.output.%d", itemIndex, partIndex),
							url:  imageURL,
						})
					}
					partIndex++
					return true
				})
			case output.Type == gjson.String:
				inner := output.String()
				if strings.Contains(inner, "image") && gjson.Valid(inner) && gjson.Parse(inner).IsArray() {
					stringOutputs = append(stringOutputs, stringOutputTarget{
						path:  fmt.Sprintf("input.%d.output", itemIndex),
						inner: inner,
					})
				}
			}
		}
		itemIndex++
		return true
	})
	out, replaced := applyTargets(body, targets, "input_text", resolve)
	for _, stringOutput := range stringOutputs {
		modified, count := replaceResponsesOutputParts([]byte(stringOutput.inner), resolve)
		if count == 0 {
			continue
		}
		if updated, errSet := sjson.SetBytes(out, stringOutput.path, string(modified)); errSet == nil {
			out = updated
			replaced += count
		}
	}
	return out, replaced
}

// geminiImagePartURL reports whether the part is an image part (inlineData or
// fileData with an image MIME type, in camelCase or snake_case), returning a
// URL for the vision call. "" with ok=true means unresolvable image.
func geminiImagePartURL(part gjson.Result) (string, bool) {
	inline := part.Get("inlineData")
	if !inline.Exists() {
		inline = part.Get("inline_data")
	}
	if inline.IsObject() {
		mime := inline.Get("mimeType").String()
		if mime == "" {
			mime = inline.Get("mime_type").String()
		}
		if !strings.HasPrefix(strings.ToLower(mime), "image/") {
			return "", false
		}
		data := inline.Get("data").String()
		if data == "" {
			return "", true
		}
		return "data:" + mime + ";base64," + data, true
	}
	file := part.Get("fileData")
	if !file.Exists() {
		file = part.Get("file_data")
	}
	if file.IsObject() {
		mime := file.Get("mimeType").String()
		if mime == "" {
			mime = file.Get("mime_type").String()
		}
		if !strings.HasPrefix(strings.ToLower(mime), "image/") {
			return "", false
		}
		uri := file.Get("fileUri").String()
		if uri == "" {
			uri = file.Get("file_uri").String()
		}
		return uri, true
	}
	return "", false
}

// replaceGeminiImages replaces image parts in Gemini generateContent requests.
// Non-image inline/file data (audio, video, documents) is left untouched.
func replaceGeminiImages(body []byte, resolve resolveText) ([]byte, int) {
	contents := gjson.GetBytes(body, "contents")
	if !contents.IsArray() {
		return body, 0
	}
	targets := make([]imageTarget, 0, 4)
	contentIndex := 0
	contents.ForEach(func(_, content gjson.Result) bool {
		parts := content.Get("parts")
		if parts.IsArray() {
			partIndex := 0
			parts.ForEach(func(_, part gjson.Result) bool {
				if imageURL, ok := geminiImagePartURL(part); ok {
					targets = append(targets, imageTarget{
						path: fmt.Sprintf("contents.%d.parts.%d", contentIndex, partIndex),
						url:  imageURL,
					})
				}
				partIndex++
				return true
			})
		}
		contentIndex++
		return true
	})
	return applyTargets(body, targets, "", resolve)
}
