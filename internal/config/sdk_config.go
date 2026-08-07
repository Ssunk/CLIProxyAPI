// Package config provides configuration management for the CLI Proxy API server.
// It handles loading and parsing YAML configuration files, and provides structured
// access to application settings including server port, authentication directory,
// debug settings, proxy configuration, and API keys.
package config

// SDKConfig represents the application's configuration, loaded from a YAML file.
type SDKConfig struct {
	// ProxyURL is the URL of an optional proxy server to use for outbound requests.
	ProxyURL string `yaml:"proxy-url" json:"proxy-url"`

	// DisableImageGeneration controls whether the built-in image_generation tool is injected/allowed.
	//
	// Supported values:
	//   - false (default): image_generation is enabled everywhere (normal behavior).
	//   - true: image_generation is disabled everywhere. The server stops injecting it, removes it from request payloads,
	//     and returns 404 for /v1/images/generations and /v1/images/edits.
	//   - "chat": disable image_generation injection for all non-images endpoints (e.g. /v1/responses, /v1/chat/completions),
	//     while keeping /v1/images/generations and /v1/images/edits enabled and preserving image_generation there.
	//   - "passthrough": do not modify the tool list on non-images endpoints — keep image_generation if the client
	//     sent it and do not inject it otherwise; on /v1/images/generations and /v1/images/edits behave like "chat".
	DisableImageGeneration DisableImageGenerationMode `yaml:"disable-image-generation" json:"disable-image-generation"`

	// GPTImage2BaseModel sets the base (mainline) model used by the legacy hosted
	// image_generation tool path when a Codex image request is not proxied directly
	// through the Image API.
	//
	// The value must start with "gpt-" (case-insensitive). If empty or invalid, the
	// default base model ("gpt-5.4-mini") is used.
	GPTImage2BaseModel string `yaml:"gpt-image-2-base-model,omitempty" json:"gpt-image-2-base-model,omitempty"`

	// VideoResultAuthCacheTTL controls how long video IDs stay pinned to the credential
	// that created them. Accepts duration strings like "30m" or "3h".
	// Empty or invalid values use the default 3h.
	VideoResultAuthCacheTTL string `yaml:"video-result-auth-cache-ttl,omitempty" json:"video-result-auth-cache-ttl,omitempty"`

	// ForceModelPrefix requires explicit model prefixes (e.g., "teamA/gemini-3-pro-preview")
	// to target prefixed credentials. When false, unprefixed model requests may use prefixed
	// credentials as well.
	ForceModelPrefix bool `yaml:"force-model-prefix" json:"force-model-prefix"`

	// RequestLog enables or disables detailed request logging functionality.
	RequestLog bool `yaml:"request-log" json:"request-log"`

	// APIKeys is a list of keys for authenticating clients to this proxy server.
	APIKeys []string `yaml:"api-keys" json:"api-keys"`

	// PassthroughHeaders controls whether upstream response headers are forwarded to downstream clients.
	// Default is false (disabled).
	PassthroughHeaders bool `yaml:"passthrough-headers" json:"passthrough-headers"`

	// Streaming configures server-side streaming behavior (keep-alives and safe bootstrap retries).
	Streaming StreamingConfig `yaml:"streaming" json:"streaming"`

	// NonStreamKeepAliveInterval controls how often blank lines are emitted for non-streaming responses.
	// <= 0 disables keep-alives. Value is in seconds.
	NonStreamKeepAliveInterval int `yaml:"nonstream-keepalive-interval,omitempty" json:"nonstream-keepalive-interval,omitempty"`

	// VisionFallback configures automatic image-to-text fallback for blacklisted
	// non-vision models. When a model listed in Models receives a request
	// containing images, each image part is replaced by the text description
	// produced by a vision-capable model before the request is forwarded upstream.
	// Disabled when absent or enabled=false.
	VisionFallback VisionFallbackConfig `yaml:"vision-fallback" json:"vision-fallback"`
}

// VisionFallbackConfig configures automatic image-to-text fallback for models
// that do not support image input. When a blacklisted model receives a request
// containing images, each image part is replaced by the text description
// produced by a vision-capable model before the request is forwarded upstream.
type VisionFallbackConfig struct {
	// Enabled toggles vision fallback processing.
	Enabled bool `yaml:"enabled" json:"enabled"`

	// BaseURL is the base URL of an OpenAI-compatible chat completions endpoint
	// used to describe images (e.g. "https://api.openai.com/v1").
	BaseURL string `yaml:"base-url" json:"base-url"`

	// APIKey authenticates requests to the vision endpoint. Plaintext in config,
	// matching the existing gemini-api-key / claude-api-key convention.
	APIKey string `yaml:"api-key" json:"api-key"`

	// Model is the vision-capable model name used to describe images.
	Model string `yaml:"model" json:"model"`

	// Prompt is the instruction sent to the vision model with each image.
	// Default: "Describe the image content in detail, including any visible text,
	// objects, and layout. Output only the description."
	Prompt string `yaml:"prompt,omitempty" json:"prompt,omitempty"`

	// TimeoutSeconds bounds each vision API request. Default 30.
	TimeoutSeconds int `yaml:"timeout-seconds,omitempty" json:"timeout-seconds,omitempty"`

	// MaxTokens caps the vision model description length. Default 512.
	MaxTokens int `yaml:"max-tokens,omitempty" json:"max-tokens,omitempty"`

	// Models lists model names that trigger fallback when the request contains
	// images. Matched case-insensitively against both the client-requested model
	// and the routed upstream model.
	Models []string `yaml:"models" json:"models"`
}

// StreamingConfig holds server streaming behavior configuration.
type StreamingConfig struct {
	// KeepAliveSeconds controls how often the server emits SSE heartbeats (": keep-alive\n\n").
	// <= 0 disables keep-alives. Default is 0.
	KeepAliveSeconds int `yaml:"keepalive-seconds,omitempty" json:"keepalive-seconds,omitempty"`

	// BootstrapRetries controls how many times the server may retry a streaming request before any bytes are sent,
	// to allow auth rotation / transient recovery.
	// <= 0 disables bootstrap retries. Default is 0.
	BootstrapRetries int `yaml:"bootstrap-retries,omitempty" json:"bootstrap-retries,omitempty"`
}
