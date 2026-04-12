package scavenger

import (
        "fmt"
        "testing"
        "time"
)

func TestAddAPI(t *testing.T) {
        tracker := NewFreeTierTracker(0)
        tracker.AddAPI("test-api", 100, 1)
        if tracker.Remaining("test-api") != 100 {
                t.Fatalf("expected 100, got %d", tracker.Remaining("test-api"))
        }
}

func TestUse(t *testing.T) {
        tracker := NewFreeTierTracker(0)
        tracker.AddAPI("test-api", 5, 1)
        for i := 0; i < 5; i++ {
                if !tracker.Use("test-api") { t.Fatalf("use %d should succeed", i) }
        }
        if tracker.Use("test-api") { t.Fatal("6th use should fail") }
        if tracker.Remaining("test-api") != 0 { t.Fatalf("remaining should be 0") }
}

func TestReset(t *testing.T) {
        tracker := NewFreeTierTracker(0)
        tracker.AddAPI("test-api", 10, 1)
        for i := 0; i < 10; i++ { tracker.Use("test-api") }
        tracker.lastReset = tracker.lastReset.Add(-25 * time.Hour)
        tracker.ResetIfNeeded()
        if tracker.Remaining("test-api") != 10 { t.Fatalf("after reset, remaining should be 10") }
}

func TestShouldScavenge(t *testing.T) {
        tracker := NewFreeTierTracker(0)
        tracker.AddAPI("free", 1000, 10)
        if !tracker.ShouldScavenge() { t.Fatal("should scavenge with 1000 remaining") }
        for i := 0; i < 991; i++ { tracker.Use("free") }
        if tracker.ShouldScavenge() { t.Fatal("should not scavenge with <10 remaining") }
}

func TestTimeUntilReset(t *testing.T) {
        tracker := NewFreeTierTracker(0)
        dur := tracker.TimeUntilReset()
        if dur <= 0 || dur > 24*time.Hour { t.Fatalf("unexpected: %v", dur) }
}

func TestScavengeRemaining(t *testing.T) {
        tracker := NewFreeTierTracker(0)
        tracker.AddAPI("ideation-api", 20, 10)
        for i := 0; i < 10; i++ { tracker.Use("ideation-api") }
        callCount := 0
        results := tracker.ScavengeRemaining(func(api string) (string, error) {
                callCount++
                return fmt.Sprintf("idea-%d", callCount), nil
        })
        if len(results) == 0 { t.Fatal("should have scavenged") }
        if results[0].Calls != 10 { t.Fatalf("expected 10, got %d", results[0].Calls) }
}

func TestStatus(t *testing.T) {
        tracker := NewFreeTierTracker(0)
        tracker.AddAPI("test", 100, 1)
        tracker.Use("test")
        status := tracker.Status()
        if status == "" { t.Fatal("status should not be empty") }
}

// --- Additional edge case tests ---

func TestRemainingUnknownAPI(t *testing.T) {
        tracker := NewFreeTierTracker(0)
        if tracker.Remaining("nonexistent") != 0 {
                t.Fatalf("unknown API should return 0, got %d", tracker.Remaining("nonexistent"))
        }
}

func TestUseUnknownAPI(t *testing.T) {
        tracker := NewFreeTierTracker(0)
        if tracker.Use("nonexistent") {
                t.Fatal("Use on unknown API should return false")
        }
}

func TestResetNotTriggered(t *testing.T) {
        tracker := NewFreeTierTracker(0)
        tracker.lastReset = time.Now() // prevent reset from triggering
        tracker.AddAPI("test", 5, 1)
        tracker.Use("test")
        tracker.Use("test")
        // Don't manipulate time - reset should not trigger
        tracker.ResetIfNeeded()
        if tracker.Remaining("test") != 3 {
                t.Fatalf("expected 3 remaining, got %d", tracker.Remaining("test"))
        }
}

func TestAddAPIMultiple(t *testing.T) {
        tracker := NewFreeTierTracker(0)
        tracker.AddAPI("api-a", 100, 1)
        tracker.AddAPI("api-b", 200, 2)
        tracker.AddAPI("api-c", 300, 3)
        if tracker.Remaining("api-a") != 100 { t.Fatal("api-a wrong") }
        if tracker.Remaining("api-b") != 200 { t.Fatal("api-b wrong") }
        if tracker.Remaining("api-c") != 300 { t.Fatal("api-c wrong") }
}

func TestShouldScavengeMultipleAPIs(t *testing.T) {
        tracker := NewFreeTierTracker(0)
        tracker.AddAPI("exhausted", 5, 1)
        tracker.AddAPI("plenty", 1000, 2)
        // Exhaust first
        for i := 0; i < 5; i++ { tracker.Use("exhausted") }
        // Second has plenty, so should still scavenge
        if !tracker.ShouldScavenge() { t.Fatal("should scavenge with one API having plenty") }
        // Exhaust both
        for i := 0; i < 990; i++ { tracker.Use("plenty") }
        if tracker.ShouldScavenge() { t.Fatal("should not scavenge when all APIs exhausted") }
}

func TestShouldScavengeEmpty(t *testing.T) {
        tracker := NewFreeTierTracker(0)
        if tracker.ShouldScavenge() { t.Fatal("empty tracker should not scavenge") }
}

func TestScavengeRemainingWithErrors(t *testing.T) {
        tracker := NewFreeTierTracker(0)
        tracker.AddAPI("flaky-api", 20, 10)
        callCount := 0
        results := tracker.ScavengeRemaining(func(api string) (string, error) {
                callCount++
                if callCount%2 == 0 {
                        return "", fmt.Errorf("simulated error")
                }
                return "ok", nil
        })
        if len(results) == 0 { t.Fatal("should have scavenged") }
        // Some calls should have failed
        if results[0].Ideas == results[0].Calls {
                t.Fatal("expected some calls to fail")
        }
}

func TestScavengeRemainingNoAPIs(t *testing.T) {
        tracker := NewFreeTierTracker(0)
        results := tracker.ScavengeRemaining(func(api string) (string, error) {
                return "unused", nil
        })
        if len(results) != 0 { t.Fatalf("expected 0 results for empty tracker, got %d", len(results)) }
}

func TestScavengeRemainingExhaustedAPI(t *testing.T) {
        tracker := NewFreeTierTracker(0)
        tracker.lastReset = time.Now() // prevent reset from triggering
        tracker.AddAPI("used-up", 2, 1)
        tracker.Use("used-up")
        tracker.Use("used-up")
        results := tracker.ScavengeRemaining(func(api string) (string, error) {
                return "should not be called", nil
        })
        // Exhausted API should be skipped (remaining <= 0 → continue)
        if len(results) != 0 { t.Fatalf("expected 0 results for exhausted API, got %d", len(results)) }
}

func TestTimeUntilResetPastHour(t *testing.T) {
        // If reset hour is in the past today, should point to tomorrow
        now := time.Now().UTC()
        pastHour := (now.Hour() + 23) % 24 // yesterday's hour
        tracker := NewFreeTierTracker(pastHour)
        dur := tracker.TimeUntilReset()
        // Should be less than 24 hours
        if dur > 24*time.Hour {
                t.Fatalf("expected < 24h, got %v", dur)
        }
        // Should be positive
        if dur <= 0 {
                t.Fatalf("expected positive duration, got %v", dur)
        }
}

func TestStatusMultipleAPIs(t *testing.T) {
        tracker := NewFreeTierTracker(3)
        tracker.AddAPI("openai", 500, 1)
        tracker.AddAPI("anthropic", 1000, 2)
        for i := 0; i < 250; i++ { tracker.Use("openai") }
        for i := 0; i < 750; i++ { tracker.Use("anthropic") }
        status := tracker.Status()
        // Should mention both APIs
        if len(status) < 50 {
                t.Fatalf("status too short: %q", status)
        }
}
