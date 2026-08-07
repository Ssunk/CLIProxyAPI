package vision

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/tidwall/gjson"
)

func visionCfg(enabled bool, model string, models ...string) *config.VisionFallbackConfig {
	return &config.VisionFallbackConfig{
		Enabled: enabled,
		APIKey:  "test-key",
		Model:   model,
		Models:  models,
	}
}

// visionServer returns an httptest.Server serving chat completions with the
// given content and status, plus a counter of requests received.
func visionServer(content string, status int) (*httptest.Server, *int32) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.Header().Set("Content-Type", "application/json")
		if status != http.StatusOK {
			w.WriteHeader(status)
			_, _ = w.Write([]byte(`{"error":"boom"}`))
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{
				map[string]any{"message": map[string]any{"content": content}},
			},
		})
	}))
	return srv, &calls
}

func TestApplyOpenAIReplacesImageInPlace(t *testing.T) {
	cfg := visionCfg(true, "gpt-5-mini", "blacklist-model")
	srv, _ := visionServer("a red car", http.StatusOK)
	defer srv.Close()
	cfg.BaseURL = srv.URL

	body := []byte(`{"model":"blacklist-model","messages":[
		{"role":"user","content":[
			{"type":"text","text":"what is this?"},
			{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}
		]}
	]}`)

	out := Apply(context.Background(), body, "openai", "blacklist-model", "blacklist-model", cfg, srv.Client())

	image := gjson.GetBytes(out, "messages.0.content.1")
	if image.Get("type").String() != "text" {
		t.Fatalf("image part not replaced with text, got type %q", image.Get("type").String())
	}
	if got := image.Get("text").String(); got != "[Image: a red car]" {
		t.Fatalf("unexpected description text %q", got)
	}
	if _, ok := image.Get("image_url").Value().(map[string]any); ok {
		t.Fatalf("image_url residue left on replaced part")
	}
	neighbor := gjson.GetBytes(out, "messages.0.content.0")
	if neighbor.Get("text").String() != "what is this?" {
		t.Fatalf("neighboring text part was modified: %s", neighbor.Raw)
	}
	if len(gjson.GetBytes(out, "messages.0.content").Array()) != 2 {
		t.Fatalf("content array length changed: expected 2")
	}
}

func TestApplyOpenAIHTTPURLImage(t *testing.T) {
	cfg := visionCfg(true, "gpt-5-mini", "blacklist-model")
	srv, _ := visionServer("a logo", http.StatusOK)
	defer srv.Close()
	cfg.BaseURL = srv.URL

	body := []byte(`{"model":"blacklist-model","messages":[
		{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}
	]}`)

	out := Apply(context.Background(), body, "openai", "blacklist-model", "blacklist-model", cfg, srv.Client())
	if got := gjson.GetBytes(out, "messages.0.content.0.text").String(); got != "[Image: a logo]" {
		t.Fatalf("unexpected description text %q", got)
	}
}

func TestApplyClaudeBase64Image(t *testing.T) {
	cfg := visionCfg(true, "gpt-5-mini", "claude-model")
	srv, _ := visionServer("a chart", http.StatusOK)
	defer srv.Close()
	cfg.BaseURL = srv.URL

	body := []byte(`{"model":"claude-model","messages":[
		{"role":"user","content":[
			{"type":"image","source":{"type":"base64","media_type":"image/png","data":"QUFBQQ=="}}
		]}
	]}`)

	out := Apply(context.Background(), body, "claude", "claude-model", "claude-model", cfg, srv.Client())
	part := gjson.GetBytes(out, "messages.0.content.0")
	if part.Get("type").String() != "text" || part.Get("text").String() != "[Image: a chart]" {
		t.Fatalf("unexpected replacement part: %s", part.Raw)
	}
}

func TestApplyGeminiInlineData(t *testing.T) {
	cfg := visionCfg(true, "gpt-5-mini", "gemini-model")
	srv, _ := visionServer("a landscape", http.StatusOK)
	defer srv.Close()
	cfg.BaseURL = srv.URL

	// Camel-case variant.
	body := []byte(`{"model":"gemini-model","contents":[
		{"parts":[{"text":"look"},{"inlineData":{"mimeType":"image/png","data":"QUFBQQ=="}}]}
	]}`)

	out := Apply(context.Background(), body, "gemini", "gemini-model", "gemini-model", cfg, srv.Client())
	part := gjson.GetBytes(out, "contents.0.parts.1")
	if part.Get("text").String() != "[Image: a landscape]" {
		t.Fatalf("unexpected replacement part: %s", part.Raw)
	}
	neighbor := gjson.GetBytes(out, "contents.0.parts.0")
	if neighbor.Get("text").String() != "look" {
		t.Fatalf("neighboring part modified: %s", neighbor.Raw)
	}

	// Snake-case variant.
	bodySnake := []byte(`{"model":"gemini-model","contents":[
		{"parts":[{"inline_data":{"mime_type":"image/jpeg","data":"QkJC"}}]}
	]}`)
	outSnake := Apply(context.Background(), bodySnake, "gemini", "gemini-model", "gemini-model", cfg, srv.Client())
	if got := gjson.GetBytes(outSnake, "contents.0.parts.0.text").String(); got != "[Image: a landscape]" {
		t.Fatalf("snake-case inline_data not replaced, got %q", got)
	}
}

func TestApplyCodexInputImage(t *testing.T) {
	cfg := visionCfg(true, "gpt-5-mini", "codex-model")
	srv, _ := visionServer("a diagram", http.StatusOK)
	defer srv.Close()
	cfg.BaseURL = srv.URL

	// Top-level input_image plus a nested input_image inside a message.
	body := []byte(`{"model":"codex-model","input":[
		{"type":"input_image","image_url":"https://example.com/top.png"},
		{"type":"message","role":"user","content":[
			{"type":"input_text","text":"hi"},
			{"type":"input_image","image_url":{"url":"https://example.com/nested.png"}}
		]}
	]}`)

	out := Apply(context.Background(), body, "codex", "codex-model", "codex-model", cfg, srv.Client())

	top := gjson.GetBytes(out, "input.0")
	if top.Get("type").String() != "input_text" || top.Get("text").String() != "[Image: a diagram]" {
		t.Fatalf("top-level input_image not replaced: %s", top.Raw)
	}
	nested := gjson.GetBytes(out, "input.1.content.1")
	if nested.Get("type").String() != "input_text" || nested.Get("text").String() != "[Image: a diagram]" {
		t.Fatalf("nested input_image not replaced: %s", nested.Raw)
	}
	neighbor := gjson.GetBytes(out, "input.1.content.0")
	if neighbor.Get("text").String() != "hi" {
		t.Fatalf("neighboring input_text modified: %s", neighbor.Raw)
	}
}

func TestApplyFileIDOnlySkipped(t *testing.T) {
	cfg := visionCfg(true, "gpt-5-mini", "codex-model")
	srv, calls := visionServer("x", http.StatusOK)
	defer srv.Close()
	cfg.BaseURL = srv.URL

	body := []byte(`{"model":"codex-model","input":[
		{"type":"input_image","file_id":"file-123"}
	]}`)

	out := Apply(context.Background(), body, "codex", "codex-model", "codex-model", cfg, srv.Client())
	if got := gjson.GetBytes(out, "input.0.type").String(); got != "input_image" {
		t.Fatalf("file_id-only input_image should be left alone, got %q", got)
	}
	if *calls != 0 {
		t.Fatalf("vision endpoint called for unresolvable file_id image")
	}
}

func TestApplyNoImagesReturnsOriginal(t *testing.T) {
	cfg := visionCfg(true, "gpt-5-mini", "blacklist-model")
	srv, _ := visionServer("x", http.StatusOK)
	defer srv.Close()
	cfg.BaseURL = srv.URL

	body := []byte(`{"model":"blacklist-model","messages":[
		{"role":"user","content":[{"type":"text","text":"hello"}]}
	]}`)

	out := Apply(context.Background(), body, "openai", "blacklist-model", "blacklist-model", cfg, srv.Client())
	if !bytes.Equal(out, body) {
		t.Fatalf("body with no images must be returned unchanged")
	}
}

func TestApplyBlacklistMatchOnRequestedOrRouted(t *testing.T) {
	srv, _ := visionServer("x", http.StatusOK)
	defer srv.Close()
	cfg := visionCfg(true, "gpt-5-mini", "blacklist-model")
	cfg.BaseURL = srv.URL
	body := []byte(`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}`)

	// Requested model matches, routed model does not.
	if out := Apply(context.Background(), body, "openai", "blacklist-model", "other-model", cfg, srv.Client()); bytes.Equal(out, body) {
		t.Fatalf("requested model in blacklist should trigger fallback")
	}
	// Routed model matches, requested model does not.
	if out := Apply(context.Background(), body, "openai", "other-model", "blacklist-model", cfg, srv.Client()); bytes.Equal(out, body) {
		t.Fatalf("routed model in blacklist should trigger fallback")
	}
	// Neither matches.
	if out := Apply(context.Background(), body, "openai", "other-model", "other-model", cfg, srv.Client()); !bytes.Equal(out, body) {
		t.Fatalf("non-blacklisted models must pass through unchanged")
	}
}

func TestApplyBlacklistCaseInsensitive(t *testing.T) {
	srv, _ := visionServer("x", http.StatusOK)
	defer srv.Close()
	cfg := visionCfg(true, "gpt-5-mini", "CLAUDE-Sonnet-4-5 ")
	cfg.BaseURL = srv.URL
	body := []byte(`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}`)

	if out := Apply(context.Background(), body, "openai", "claude-sonnet-4-5", "claude-sonnet-4-5", cfg, srv.Client()); bytes.Equal(out, body) {
		t.Fatalf("blacklist matching must be case-insensitive and trim whitespace")
	}
}

func TestApplyDisabledReturnsOriginal(t *testing.T) {
	srv, _ := visionServer("x", http.StatusOK)
	defer srv.Close()
	cfg := visionCfg(false, "gpt-5-mini", "blacklist-model")
	cfg.BaseURL = srv.URL
	body := []byte(`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}`)

	if out := Apply(context.Background(), body, "openai", "blacklist-model", "blacklist-model", cfg, srv.Client()); !bytes.Equal(out, body) {
		t.Fatalf("disabled config must leave body unchanged")
	}
}

func TestApplyVisionCallAndParse(t *testing.T) {
	cfg := visionCfg(true, "gpt-5-mini", "blacklist-model")
	var gotURL, gotAuth, gotBody string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotURL = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		b := make([]byte, 0, 4096)
		buf := bytes.NewBuffer(b)
		_, _ = buf.ReadFrom(r.Body)
		gotBody = buf.String()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"a red car"}}]}`))
	}))
	defer srv.Close()
	cfg.BaseURL = srv.URL

	body := []byte(`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,QUFBQQ=="}}]}]}`)
	out := Apply(context.Background(), body, "openai", "blacklist-model", "blacklist-model", cfg, srv.Client())

	if gotURL != "/chat/completions" {
		t.Fatalf("unexpected vision endpoint path %q", gotURL)
	}
	if gotAuth != "Bearer test-key" {
		t.Fatalf("unexpected auth header %q", gotAuth)
	}
	if !strings.Contains(gotBody, "data:image/png;base64,QUFBQQ==") {
		t.Fatalf("vision request missing data URI: %s", gotBody)
	}
	if !strings.Contains(gotBody, "gpt-5-mini") {
		t.Fatalf("vision request missing model: %s", gotBody)
	}
	if got := gjson.GetBytes(out, "messages.0.content.0.text").String(); got != "[Image: a red car]" {
		t.Fatalf("unexpected description text %q", got)
	}
}

func TestApplyContentPartsArrayResponse(t *testing.T) {
	cfg := visionCfg(true, "gpt-5-mini", "blacklist-model")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":[{"type":"text","text":"first line"},{"type":"text","text":"second line"}]}}]}`))
	}))
	defer srv.Close()
	cfg.BaseURL = srv.URL

	body := []byte(`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,QUFBQQ=="}}]}]}`)
	out := Apply(context.Background(), body, "openai", "blacklist-model", "blacklist-model", cfg, srv.Client())

	want := "[Image: first line\nsecond line]"
	if got := gjson.GetBytes(out, "messages.0.content.0.text").String(); got != want {
		t.Fatalf("content-parts response not joined, got %q want %q", got, want)
	}
}

func TestApplyDedupsIdenticalImages(t *testing.T) {
	cfg := visionCfg(true, "gpt-5-mini", "blacklist-model")
	srv, calls := visionServer("same image", http.StatusOK)
	defer srv.Close()
	cfg.BaseURL = srv.URL

	body := []byte(`{"messages":[{"role":"user","content":[
		{"type":"image_url","image_url":{"url":"data:image/png;base64,QUFBQQ=="}},
		{"type":"image_url","image_url":{"url":"data:image/png;base64,QUFBQQ=="}}
	]}]}`)

	out := Apply(context.Background(), body, "openai", "blacklist-model", "blacklist-model", cfg, srv.Client())
	if *calls != 1 {
		t.Fatalf("identical images should trigger one vision call, got %d", *calls)
	}
	if got := gjson.GetBytes(out, "messages.0.content.1.text").String(); got != "[Image: same image]" {
		t.Fatalf("dedup replacement missing: %q", got)
	}
}

func TestApplyFailOpenOnVisionError(t *testing.T) {
	cfg := visionCfg(true, "gpt-5-mini", "blacklist-model")
	srv, _ := visionServer("", http.StatusInternalServerError)
	defer srv.Close()
	cfg.BaseURL = srv.URL

	body := []byte(`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,QUFBQQ=="}}]}]}`)
	out := Apply(context.Background(), body, "openai", "blacklist-model", "blacklist-model", cfg, srv.Client())
	if !bytes.Equal(out, body) {
		t.Fatalf("vision failure must fail open and return original body")
	}
}
