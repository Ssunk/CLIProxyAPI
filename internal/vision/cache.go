package vision

import (
	"crypto/sha256"
	"encoding/hex"
	"sync"
	"time"
)

const (
	descriptionCacheTTL  = 3 * time.Hour
	cacheCleanupInterval = 10 * time.Minute
	maxCacheEntries      = 512
)

type cacheEntry struct {
	description string
	timestamp   time.Time
}

// descriptionCache stores image descriptions keyed by vision model, prompt,
// and image payload so images resent across conversation turns are only
// described once. Entries use a sliding TTL.
type descriptionCache struct {
	mu      sync.Mutex
	entries map[string]cacheEntry
	now     func() time.Time
}

var defaultCache = newDescriptionCache()

func newDescriptionCache() *descriptionCache {
	return &descriptionCache{
		entries: make(map[string]cacheEntry),
		now:     time.Now,
	}
}

// cacheKey hashes the inputs so full base64 image payloads never stay in
// memory; only the resulting description is stored.
func cacheKey(visionModel, prompt, imageURL string) string {
	sum := sha256.Sum256([]byte(visionModel + "\x1f" + prompt + "\x1f" + imageURL))
	return hex.EncodeToString(sum[:])
}

// Get returns the cached description and refreshes its timestamp (sliding TTL).
func (c *descriptionCache) Get(key string) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	entry, ok := c.entries[key]
	if !ok {
		return "", false
	}
	if c.now().Sub(entry.timestamp) > descriptionCacheTTL {
		delete(c.entries, key)
		return "", false
	}
	entry.timestamp = c.now()
	c.entries[key] = entry
	return entry.description, true
}

// Put stores a description, evicting expired entries first and the oldest
// entry when the cache is still at capacity.
func (c *descriptionCache) Put(key, description string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, exists := c.entries[key]; !exists && len(c.entries) >= maxCacheEntries {
		c.evictLocked()
	}
	c.entries[key] = cacheEntry{description: description, timestamp: c.now()}
}

func (c *descriptionCache) evictLocked() {
	now := c.now()
	for key, entry := range c.entries {
		if now.Sub(entry.timestamp) > descriptionCacheTTL {
			delete(c.entries, key)
		}
	}
	if len(c.entries) < maxCacheEntries {
		return
	}
	oldestKey := ""
	var oldestTime time.Time
	for key, entry := range c.entries {
		if oldestKey == "" || entry.timestamp.Before(oldestTime) {
			oldestKey = key
			oldestTime = entry.timestamp
		}
	}
	if oldestKey != "" {
		delete(c.entries, oldestKey)
	}
}

// cleanupLoop periodically drops expired entries. It is started once for the
// package cache on first use.
func (c *descriptionCache) cleanupLoop() {
	ticker := time.NewTicker(cacheCleanupInterval)
	defer ticker.Stop()
	for range ticker.C {
		c.mu.Lock()
		now := c.now()
		for key, entry := range c.entries {
			if now.Sub(entry.timestamp) > descriptionCacheTTL {
				delete(c.entries, key)
			}
		}
		c.mu.Unlock()
	}
}
