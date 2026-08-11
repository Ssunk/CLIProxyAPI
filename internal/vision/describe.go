package vision

import (
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// DefaultPrompt is the built-in image description prompt, overridable via
// vision-fallback.prompt.
const DefaultPrompt = "You are describing an image for a text-only AI model that cannot see images. " +
	"Describe this image accurately and completely. Include: any visible text transcribed verbatim; " +
	"the kind of content (photo, screenshot, diagram, chart, code, document, UI); " +
	"layout, structure, and notable visual details; and anything needed to answer questions about the image. " +
	"Be thorough but do not speculate beyond what is visible. Output only the description."

// BuildVisionRequest builds a non-streaming OpenAI chat-completions request
// for the vision model. maxTokens <= 0 omits max_tokens.
func BuildVisionRequest(visionModel, prompt, imageURL string, maxTokens int) []byte {
	body := []byte(`{"model":"","stream":false,"messages":[{"role":"user","content":[{"type":"text","text":""},{"type":"image_url","image_url":{"url":""}}]}]}`)
	body, _ = sjson.SetBytes(body, "model", visionModel)
	body, _ = sjson.SetBytes(body, "messages.0.content.0.text", prompt)
	body, _ = sjson.SetBytes(body, "messages.0.content.1.image_url.url", imageURL)
	if maxTokens > 0 {
		body, _ = sjson.SetBytes(body, "max_tokens", maxTokens)
	}
	return body
}

// ParseVisionResponse extracts the assistant text from an OpenAI
// chat-completions response. It returns "" when absent or empty.
func ParseVisionResponse(body []byte) string {
	content := gjson.GetBytes(body, "choices.0.message.content")
	switch {
	case content.Type == gjson.String:
		return strings.TrimSpace(content.String())
	case content.IsArray():
		parts := make([]string, 0, 4)
		content.ForEach(func(_, item gjson.Result) bool {
			if text := item.Get("text"); text.Type == gjson.String {
				if trimmed := strings.TrimSpace(text.String()); trimmed != "" {
					parts = append(parts, trimmed)
				}
			}
			return true
		})
		return strings.Join(parts, "\n")
	}
	return ""
}
