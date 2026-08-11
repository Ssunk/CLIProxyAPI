package handlers

import (
	"fmt"
	"net/http"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/constant"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/vision"
	"golang.org/x/net/context"
)

// applyVisionFallback replaces image blocks with text descriptions when the
// requested model does not accept image input. It returns rawJSON unchanged
// whenever the feature does not engage. The result is used only as the
// executor payload; opts.OriginalRequest keeps the untouched request so
// session identity, affinity, and replay caches stay stable.
func (h *BaseAPIHandler) applyVisionFallback(ctx context.Context, entryProtocol, normalizedModel string, providers []string, rawJSON []byte, execOptions modelExecutionOptions) []byte {
	if h == nil || h.Cfg == nil || len(rawJSON) == 0 {
		return rawJSON
	}
	cfg := h.Cfg.VisionFallback
	if !cfg.Enabled || cfg.Model == "" {
		return rawJSON
	}
	// Internal model executions (including our own describe calls and plugin
	// host callbacks) are never transformed; this also hard-stops recursion.
	if execOptions.InternalSource {
		return rawJSON
	}
	switch entryProtocol {
	case constant.Claude, constant.OpenAI, constant.OpenaiResponse, constant.Gemini:
	default:
		return rawJSON
	}
	describe := func(describeCtx context.Context, body []byte) ([]byte, error) {
		resp, errMsg := h.ExecuteModel(describeCtx, ModelExecutionRequest{
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
		Providers: providers,
		Config:    cfg,
		Describe:  describe,
	})
}
