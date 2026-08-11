package handlers

import (
	"fmt"
	"net/http"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/constant"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/vision"
	"golang.org/x/net/context"
)

type contextWithoutValues struct {
	context.Context
}

func (contextWithoutValues) Value(any) any {
	return nil
}

func isolatedExecutionContext(ctx context.Context) context.Context {
	if ctx == nil {
		return context.Background()
	}
	return contextWithoutValues{Context: ctx}
}

func (h *BaseAPIHandler) visionFallbackEnabled(execOptions modelExecutionOptions) bool {
	return h != nil && h.Cfg != nil && h.Cfg.VisionFallback.Enabled && h.Cfg.VisionFallback.Model != "" && !execOptions.InternalSource
}

// applyVisionFallback replaces image blocks with text descriptions when the
// requested model does not accept image input. It returns rawJSON unchanged
// whenever the feature does not engage. The result is used only as the
// executor payload; opts.OriginalRequest keeps the untouched request so
// session identity, affinity, and replay caches stay stable.
func (h *BaseAPIHandler) applyVisionFallback(ctx context.Context, entryProtocol, normalizedModel, provider string, rawJSON []byte, execOptions modelExecutionOptions) []byte {
	if h == nil || h.Cfg == nil || len(rawJSON) == 0 {
		return rawJSON
	}
	cfg := h.Cfg.VisionFallback
	if !h.visionFallbackEnabled(execOptions) {
		return rawJSON
	}
	switch entryProtocol {
	case constant.Claude, constant.OpenAI, constant.OpenaiResponse, constant.Gemini:
	default:
		return rawJSON
	}
	describe := func(describeCtx context.Context, body []byte) ([]byte, error) {
		resp, errMsg := h.ExecuteModel(isolatedExecutionContext(describeCtx), ModelExecutionRequest{
			EntryProtocol: constant.OpenAI,
			Model:         cfg.Model,
			Stream:        false,
			Body:          body,
			// Explicit minimal headers keep the internal call independent of
			// the inbound client protocol (no context header inheritance).
			Headers: http.Header{"Content-Type": []string{"application/json"}},
		})
		if errMsg != nil {
			if errMsg.Error != nil {
				return nil, errMsg.Error
			}
			return nil, fmt.Errorf("vision model call failed with status %d", errMsg.StatusCode)
		}
		return resp.Body, nil
	}
	return vision.Apply(ctx, rawJSON, vision.Options{
		Format:    entryProtocol,
		Model:     normalizedModel,
		Providers: []string{provider},
		Config:    cfg,
		Describe:  describe,
	})
}
