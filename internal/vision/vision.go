// Package vision implements the vision-fallback feature: when a request that
// contains images targets a model without image input support, each image
// block is replaced in-place with a text description produced by a configured
// vision-capable model. The transformed body is used only as the upstream
// payload; callers keep the original request untouched so session identity
// and replay caches stay stable.
package vision

import (
	"errors"
	"sync"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/constant"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/registry"
	log "github.com/sirupsen/logrus"
	"golang.org/x/net/context"
	"golang.org/x/sync/singleflight"
)

const (
	imageMarkerPrefix      = "[Image content: "
	imageMarkerSuffix      = "]"
	imageUnavailableMarker = "[Image content unavailable]"
	maxImagesPerRequest    = 16
)

// DescribeFunc executes an internal non-streaming OpenAI chat-completions
// request against the configured vision model and returns the raw response body.
type DescribeFunc func(ctx context.Context, requestBody []byte) ([]byte, error)

// ModelInfoLookup resolves registry model info; nil means unknown model.
type ModelInfoLookup func(model, provider string) *registry.ModelInfo

// Options carries everything Apply needs for one request.
type Options struct {
	// Format is the source request format: "claude", "openai",
	// "openai-response", or "gemini".
	Format string
	// Model is the normalized target model (may carry a thinking suffix or a
	// credential prefix).
	Model string
	// Providers holds the resolved providers; the first entry is used as the
	// registry lookup hint.
	Providers []string
	// Config is the vision fallback configuration.
	Config config.VisionFallbackConfig
	// Describe performs the internal vision model call.
	Describe DescribeFunc
	// Lookup resolves model info; nil uses the global registry.
	Lookup ModelInfoLookup
}

var (
	cacheCleanupStart   sync.Once
	describeGroup       singleflight.Group
	errEmptyDescription = errors.New("empty image description")
)

// Apply returns the transformed body, or body unchanged when the fallback does
// not engage. It never fails the request: description errors degrade to an
// unavailable placeholder per image.
func Apply(ctx context.Context, body []byte, opts Options) []byte {
	if len(body) == 0 || opts.Describe == nil {
		return body
	}
	lookup := opts.Lookup
	if lookup == nil {
		lookup = defaultLookup
	}
	if !ShouldFallback(opts.Config, opts.Model, opts.Providers, lookup) {
		return body
	}
	if !hasImageHint(opts.Format, body) {
		return body
	}
	cacheCleanupStart.Do(func() { go defaultCache.cleanupLoop() })

	prompt := opts.Config.Prompt
	if prompt == "" {
		prompt = DefaultPrompt
	}
	visionModel := opts.Config.Model

	memo := make(map[string]string)
	described, cachedHits, failed := 0, 0, 0
	resolve := func(imageURL string) string {
		if imageURL == "" {
			failed++
			return imageUnavailableMarker
		}
		if marker, ok := memo[imageURL]; ok {
			return marker
		}
		if len(memo) >= maxImagesPerRequest {
			log.WithFields(log.Fields{
				"model":        opts.Model,
				"vision_model": visionModel,
				"limit":        maxImagesPerRequest,
			}).Warn("vision fallback: too many distinct images in one request")
			failed++
			return imageUnavailableMarker
		}
		key := cacheKey(visionModel, prompt, imageURL, opts.Config.MaxTokens)
		cacheable := cacheableImage(imageURL)
		if cacheable {
			if desc, ok := defaultCache.Get(key); ok {
				cachedHits++
				marker := imageMarkerPrefix + desc + imageMarkerSuffix
				memo[imageURL] = marker
				return marker
			}
		}
		value, errDescribe, shared := describeGroup.Do(key, func() (any, error) {
			if cacheable {
				if desc, ok := defaultCache.Get(key); ok {
					return desc, nil
				}
			}
			respBody, errDescribe := opts.Describe(ctx, BuildVisionRequest(visionModel, prompt, imageURL, opts.Config.MaxTokens))
			if errDescribe != nil {
				return "", errDescribe
			}
			desc := ParseVisionResponse(respBody)
			if desc == "" {
				return "", errEmptyDescription
			}
			if cacheable {
				defaultCache.Put(key, desc)
			}
			return desc, nil
		})
		if errDescribe != nil {
			if errDescribe == errEmptyDescription {
				log.WithFields(log.Fields{
					"model":        opts.Model,
					"vision_model": visionModel,
				}).Warn("vision fallback: empty image description")
			} else {
				log.WithFields(log.Fields{
					"model":        opts.Model,
					"vision_model": visionModel,
					"error":        errDescribe.Error(),
				}).Warn("vision fallback: image description failed")
			}
			failed++
			memo[imageURL] = imageUnavailableMarker
			return imageUnavailableMarker
		}
		desc, _ := value.(string)
		if desc == "" {
			failed++
			memo[imageURL] = imageUnavailableMarker
			return imageUnavailableMarker
		}
		if shared {
			cachedHits++
		} else {
			described++
		}
		marker := imageMarkerPrefix + desc + imageMarkerSuffix
		memo[imageURL] = marker
		return marker
	}

	out := body
	replaced := 0
	switch opts.Format {
	case constant.Claude:
		out, replaced = replaceClaudeImages(body, resolve)
	case constant.OpenAI:
		out, replaced = replaceOpenAIImages(body, resolve)
	case constant.OpenaiResponse:
		out, replaced = replaceResponsesImages(body, resolve)
	case constant.Gemini:
		out, replaced = replaceGeminiImages(body, resolve)
	default:
		return body
	}
	if replaced > 0 {
		log.WithFields(log.Fields{
			"format":       opts.Format,
			"model":        opts.Model,
			"vision_model": visionModel,
			"images":       replaced,
			"described":    described,
			"cached":       cachedHits,
			"failed":       failed,
		}).Info("vision fallback replaced image blocks")
	}
	return out
}
