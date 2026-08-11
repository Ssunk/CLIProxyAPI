package vision

import (
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/thinking"
)

// ShouldFallback reports whether the vision fallback engages for the target
// model. It engages only when the feature is enabled with a vision model and
// the target model is explicitly text-only: either its registry input
// modalities exclude images, or it matches a configured wildcard pattern.
// Unknown or empty modalities are treated as vision-capable so models without
// declared modalities are never modified.
func ShouldFallback(cfg config.VisionFallbackConfig, model string, providers []string, lookup ModelInfoLookup) bool {
	if !cfg.Enabled || strings.TrimSpace(cfg.Model) == "" {
		return false
	}
	base := strings.TrimSpace(thinking.ParseSuffix(strings.TrimSpace(model)).ModelName)
	if base == "" {
		return false
	}
	// Never transform requests that already target the vision model; this also
	// terminates recursion for the internal describe call.
	visionBase := strings.TrimSpace(thinking.ParseSuffix(strings.TrimSpace(cfg.Model)).ModelName)
	if strings.EqualFold(base, visionBase) {
		return false
	}
	if lookup == nil {
		lookup = defaultLookup
	}
	provider := ""
	if len(providers) > 0 {
		provider = providers[0]
	}
	if info := lookup(base, provider); info != nil && modalitiesExcludeImages(info.SupportedInputModalities) {
		return true
	}
	lowerBase := strings.ToLower(base)
	for _, pattern := range cfg.Models {
		if matchPattern(pattern, lowerBase) {
			return true
		}
	}
	return false
}

func defaultLookup(model, provider string) *registry.ModelInfo {
	if provider == "" {
		return registry.LookupModelInfo(model)
	}
	return registry.LookupModelInfo(model, provider)
}

// modalitiesExcludeImages mirrors the executor helper semantics
// (inputModalitiesExcludeImages): only an explicit, non-empty modality list
// that includes "text" and lacks "image" marks a model as text-only.
func modalitiesExcludeImages(modalities []string) bool {
	if len(modalities) == 0 {
		return false
	}
	hasText := false
	for _, rawModality := range modalities {
		switch strings.ToLower(strings.TrimSpace(rawModality)) {
		case "image":
			return false
		case "text":
			hasText = true
		}
	}
	return hasText
}

// matchPattern performs wildcard matching where '*' matches any substring.
// It mirrors the service-level matchWildcard helper; both pattern and value
// are expected to be lowercased already (patterns via config sanitization).
func matchPattern(pattern, value string) bool {
	if pattern == "" {
		return false
	}
	if !strings.Contains(pattern, "*") {
		return pattern == value
	}
	parts := strings.Split(pattern, "*")
	if prefix := parts[0]; prefix != "" {
		if !strings.HasPrefix(value, prefix) {
			return false
		}
		value = value[len(prefix):]
	}
	if suffix := parts[len(parts)-1]; suffix != "" {
		if !strings.HasSuffix(value, suffix) {
			return false
		}
		value = value[:len(value)-len(suffix)]
	}
	for i := 1; i < len(parts)-1; i++ {
		segment := parts[i]
		if segment == "" {
			continue
		}
		idx := strings.Index(value, segment)
		if idx < 0 {
			return false
		}
		value = value[idx+len(segment):]
	}
	return true
}
