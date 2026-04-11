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
