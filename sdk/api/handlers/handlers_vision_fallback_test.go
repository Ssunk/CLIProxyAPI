package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	coreexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdkconfig "github.com/router-for-me/CLIProxyAPI/v7/sdk/config"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginapi"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	"github.com/tidwall/gjson"
)

// visionEchoHandler returns a fake OpenAI-compatible vision endpoint.
func visionEchoHandler() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"message": map[string]any{"content": "a red car"}}},
		})
	}))
}

func TestRequestAfterAuthInterceptorVisionFallbackWithoutPlugins(t *testing.T) {
	srv := visionEchoHandler()
	defer srv.Close()

	h := &BaseAPIHandler{
		Cfg: &sdkconfig.SDKConfig{
			VisionFallback: sdkconfig.VisionFallbackConfig{
				Enabled: true,
				BaseURL: srv.URL,
				APIKey:  "test-key",
				Model:   "gpt-5-mini",
				Models:  []string{"blacklist-model"},
			},
		},
		// PluginHost is nil on purpose: vision fallback alone must enable the interceptor.
	}

	interceptor := h.requestAfterAuthInterceptor(nil, "")
	if interceptor == nil {
		t.Fatalf("requestAfterAuthInterceptor must be non-nil when vision fallback is enabled without plugins")
	}

	resp := interceptor(context.Background(), coreexecutor.RequestAfterAuthInterceptRequest{
		SourceFormat:   sdktranslator.FormatOpenAI,
		ToFormat:       sdktranslator.FormatOpenAI,
		Model:          "blacklist-model",
		RequestedModel: "blacklist-model",
		Body: []byte(`{"model":"blacklist-model","messages":[
			{"role":"user","content":[
				{"type":"text","text":"what is this?"},
				{"type":"image_url","image_url":{"url":"data:image/png;base64,QUFBQQ=="}}
			]}
		]}`),
	})

	if len(resp.Body) == 0 {
		t.Fatalf("vision fallback interceptor returned no body")
	}
	part := gjson.GetBytes(resp.Body, "messages.0.content.1")
	if part.Get("type").String() != "text" || part.Get("text").String() != "[Image: a red car]" {
		t.Fatalf("image part not replaced by vision description: %s", part.Raw)
	}
	neighbor := gjson.GetBytes(resp.Body, "messages.0.content.0")
	if neighbor.Get("text").String() != "what is this?" {
		t.Fatalf("neighboring part modified: %s", neighbor.Raw)
	}
}

func TestRequestAfterAuthInterceptorVisionRunsBeforePlugins(t *testing.T) {
	srv := visionEchoHandler()
	defer srv.Close()

	var pluginBody []byte
	host := &handlerInterceptorTestHost{
		interceptRequestAfterAuth: func(_ context.Context, req pluginapi.RequestInterceptRequest) pluginapi.RequestInterceptResponse {
			pluginBody = cloneBytes(req.Body)
			return pluginapi.RequestInterceptResponse{Body: cloneBytes(req.Body)}
		},
	}
	h := &BaseAPIHandler{
		Cfg: &sdkconfig.SDKConfig{
			VisionFallback: sdkconfig.VisionFallbackConfig{
				Enabled: true,
				BaseURL: srv.URL,
				APIKey:  "test-key",
				Model:   "gpt-5-mini",
				Models:  []string{"blacklist-model"},
			},
		},
		PluginHost: host,
	}

	interceptor := h.requestAfterAuthInterceptor(nil, "")
	if interceptor == nil {
		t.Fatalf("requestAfterAuthInterceptor must be non-nil")
	}
	resp := interceptor(context.Background(), coreexecutor.RequestAfterAuthInterceptRequest{
		SourceFormat:   sdktranslator.FormatOpenAI,
		ToFormat:       sdktranslator.FormatOpenAI,
		Model:          "blacklist-model",
		RequestedModel: "blacklist-model",
		Body: []byte(`{"model":"blacklist-model","messages":[
			{"role":"user","content":[
				{"type":"image_url","image_url":{"url":"data:image/png;base64,QUFBQQ=="}}
			]}
		]}`),
	})

	if len(resp.Body) == 0 {
		t.Fatalf("interceptor returned no body")
	}
	if len(pluginBody) == 0 {
		t.Fatalf("plugin interceptor did not run")
	}
	part := gjson.GetBytes(pluginBody, "messages.0.content.0")
	if part.Get("text").String() != "[Image: a red car]" {
		t.Fatalf("plugin interceptor should observe the vision-replaced body, got %s", part.Raw)
	}
}

func TestRequestAfterAuthInterceptorDisabled(t *testing.T) {
	h := &BaseAPIHandler{Cfg: &sdkconfig.SDKConfig{}}
	if interceptor := h.requestAfterAuthInterceptor(nil, ""); interceptor != nil {
		t.Fatalf("requestAfterAuthInterceptor must be nil when neither plugins nor vision fallback are enabled")
	}
}
