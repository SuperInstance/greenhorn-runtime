package scavenger

import (
	"fmt"
	"log"
	"sync"
	"time"
)

// FreeTierTracker monitors and maximizes free-tier API usage
type FreeTierTracker struct {
	mu         sync.Mutex
	apis       map[string]*APIBudget
	resetHour  int // UTC hour when daily quota resets
	lastReset  time.Time
}

type APIBudget struct {
	Name         string
	DailyLimit   int
	Used         int
	LastUsed     time.Time
	Priority     int    // higher = use first (free before paid)
	CallDuration time.Duration // avg time per call
}

type ScavengeResult struct {
	API      string
	Calls    int
	Ideas    int
	Duration time.Duration
}

func NewFreeTierTracker(resetHour int) *FreeTierTracker {
	return &FreeTierTracker{
		apis:      make(map[string]*APIBudget),
		resetHour: resetHour,
	}
}

func (t *FreeTierTracker) AddAPI(name string, dailyLimit int, priority int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.apis[name] = &APIBudget{
		Name:       name,
		DailyLimit: dailyLimit,
		Priority:   priority,
	}
}

func (t *FreeTierTracker) Remaining(api string) int {
	t.mu.Lock()
	defer t.mu.Unlock()
	if b, ok := t.apis[api]; ok {
		return b.DailyLimit - b.Used
	}
	return 0
}

func (t *FreeTierTracker) Use(api string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	b, ok := t.apis[api]
	if !ok || b.Used >= b.DailyLimit {
		return false
	}
	b.Used++
	b.LastUsed = time.Now()
	return true
}

func (t *FreeTierTracker) ResetIfNeeded() {
	t.mu.Lock()
	defer t.mu.Unlock()
	now := time.Now().UTC()
	if now.Hour() >= t.resetHour && now.Sub(t.lastReset) > 12*time.Hour {
		for _, b := range t.apis {
			b.Used = 0
		}
		t.lastReset = now
	}
}

// ScavengeRemaining uses up remaining free-tier calls before daily reset
// Returns how many calls were made and what they were used for
func (t *FreeTierTracker) ScavengeRemaining(useFn func(api string) (string, error)) []ScavengeResult {
	t.ResetIfNeeded()
	
	var results []ScavengeResult
	
	// Sort APIs by priority (use free tiers first)
	type apiEntry struct {
		name     string
		budget   *APIBudget
	}
	var entries []apiEntry
	t.mu.Lock()
	for name, b := range t.apis {
		entries = append(entries, apiEntry{name, b})
	}
	t.mu.Unlock()
	
	// Use remaining quota in priority order
	for _, e := range entries {
		remaining := e.budget.DailyLimit - e.budget.Used
		if remaining <= 0 {
			continue
		}
		
		// Don't use more than 50% of remaining in one scavenge pass
		toUse := remaining / 2
		if toUse == 0 {
			toUse = 1
		}
		
		start := time.Now()
		ideas := 0
		for i := 0; i < toUse; i++ {
			if !t.Use(e.name) {
				break
			}
			if _, err := useFn(e.name); err == nil {
				ideas++
			}
		}
		
		results = append(results, ScavengeResult{
			API:      e.name,
			Calls:    toUse,
			Ideas:    ideas,
			Duration: time.Since(start),
		})
		
		log.Printf("Scavenged %s: %d calls, %d results in %v", e.name, toUse, ideas, time.Since(start))
	}
	
	return results
}

// TimeUntilReset returns how long until the next quota reset
func (t *FreeTierTracker) TimeUntilReset() time.Duration {
	now := time.Now().UTC()
	next := time.Date(now.Year(), now.Month(), now.Day(), t.resetHour, 0, 0, 0, time.UTC)
	if next.Before(now) {
		next = next.Add(24 * time.Hour)
	}
	return next.Sub(now)
}

// ShouldScavenge returns true if there's enough remaining quota to justify a scavenge
func (t *FreeTierTracker) ShouldScavenge() bool {
	for _, b := range t.apis {
		if b.DailyLimit-b.Used > 10 {
			return true
		}
	}
	return false
}

// Status returns a human-readable status
func (t *FreeTierTracker) Status() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	
	s := "Free Tier Status:\n"
	for _, b := range t.apis {
		remaining := b.DailyLimit - b.Used
		pct := float64(b.Used) / float64(b.DailyLimit) * 100
		s += fmt.Sprintf("  %s: %d/%d used (%.0f%%), %d remaining\n",
			b.Name, b.Used, b.DailyLimit, pct, remaining)
	}
	s += fmt.Sprintf("  Reset in: %v\n", t.TimeUntilReset())
	return s
}
