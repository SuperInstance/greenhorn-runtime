package coordinator

import (
        "fmt"
        "strings"
        "testing"
)

func TestPostTask(t *testing.T) {
        c := NewCoordinator()
        err := c.PostTask(&Task{ID: "fence-0x50", Title: "Test task"})
        if err != nil { t.Fatal(err) }
        if c.tasks["fence-0x50"].Status != "open" { t.Fatal("should be open") }
}

func TestPostTaskNoID(t *testing.T) {
        c := NewCoordinator()
        err := c.PostTask(&Task{Title: "No ID"})
        if err == nil { t.Fatal("should error on empty ID") }
}

func TestClaimTask(t *testing.T) {
        c := NewCoordinator()
        c.PostTask(&Task{ID: "t1", Title: "Test"})
        err := c.ClaimTask("t1", "oracle1")
        if err != nil { t.Fatal(err) }
        if c.tasks["t1"].Status != "claimed" { t.Fatal("should be claimed") }
        if c.tasks["t1"].ClaimedBy != "oracle1" { t.Fatal("wrong claimant") }
}

func TestClaimAlreadyClaimed(t *testing.T) {
        c := NewCoordinator()
        c.PostTask(&Task{ID: "t1", Title: "Test"})
        c.ClaimTask("t1", "oracle1")
        err := c.ClaimTask("t1", "jetsonclaw1")
        if err == nil { t.Fatal("should error on double claim") }
}

func TestCompleteTask(t *testing.T) {
        c := NewCoordinator()
        c.PostTask(&Task{ID: "t1", Title: "Test"})
        c.ClaimTask("t1", "oracle1")
        err := c.CompleteTask("t1", "oracle1")
        if err != nil { t.Fatal(err) }
        if c.tasks["t1"].Status != "completed" { t.Fatal("should be completed") }
}

func TestCompleteNotYours(t *testing.T) {
        c := NewCoordinator()
        c.PostTask(&Task{ID: "t1", Title: "Test"})
        c.ClaimTask("t1", "oracle1")
        err := c.CompleteTask("t1", "jetsonclaw1")
        if err == nil { t.Fatal("should error completing others task") }
}

func TestAbandonTask(t *testing.T) {
        c := NewCoordinator()
        c.PostTask(&Task{ID: "t1", Title: "Test"})
        c.ClaimTask("t1", "oracle1")
        c.AbandonTask("t1", "oracle1")
        if c.tasks["t1"].Status != "open" { t.Fatal("should be open again") }
}

func TestOpenTasks(t *testing.T) {
        c := NewCoordinator()
        c.PostTask(&Task{ID: "t1", Title: "A", Difficulty: map[string]int{"oracle1": 3}})
        c.PostTask(&Task{ID: "t2", Title: "B", Difficulty: map[string]int{"oracle1": 1}})
        c.PostTask(&Task{ID: "t3", Title: "C"})
        c.ClaimTask("t3", "jetsonclaw1")
        
        open := c.OpenTasks("oracle1")
        if len(open) != 2 { t.Fatalf("expected 2 open, got %d", len(open)) }
        // Should be sorted easiest first
        if open[0].ID != "t2" { t.Fatal("easiest task should be first") }
}

func TestFleetStatus(t *testing.T) {
        c := NewCoordinator()
        c.RegisterVessel(VesselInfo{Name: "oracle1"})
        c.PostTask(&Task{ID: "t1", Title: "A"})
        c.ClaimTask("t1", "oracle1")
        status := c.FleetStatus()
        if status == "" { t.Fatal("status should not be empty") }
}

func TestRegisterVessel(t *testing.T) {
        c := NewCoordinator()
        c.RegisterVessel(VesselInfo{Name: "oracle1", Capabilities: []string{"coord"}})
        if len(c.vessels) != 1 { t.Fatal("should have 1 vessel") }
}

func TestSuggestTasks(t *testing.T) {
        c := NewCoordinator()
        c.PostTask(&Task{ID: "t1", Title: "A"})
        c.PostTask(&Task{ID: "t2", Title: "B"})
        c.PostTask(&Task{ID: "t3", Title: "C"})
        
        suggested := c.SuggestTasks("oracle1", 2)
        if len(suggested) != 2 { t.Fatalf("expected 2, got %d", len(suggested)) }
}

func TestTaskHistory(t *testing.T) {
        c := NewCoordinator()
        c.RegisterVessel(VesselInfo{Name: "oracle1"})
        c.PostTask(&Task{ID: "t1", Title: "A"})
        c.ClaimTask("t1", "oracle1")
        c.CompleteTask("t1", "oracle1")
        
        history := c.TaskHistory("oracle1")
        if len(history) != 1 { t.Fatalf("expected 1, got %d", len(history)) }
}

func TestExportJSON(t *testing.T) {
        c := NewCoordinator()
        c.PostTask(&Task{ID: "t1", Title: "A"})
        j, err := c.ExportJSON()
        if err != nil { t.Fatal(err) }
        if j == "" { t.Fatal("json should not be empty") }
}

// --- Additional edge case tests ---

func TestClaimTaskNotFound(t *testing.T) {
        c := NewCoordinator()
        err := c.ClaimTask("nonexistent", "oracle1")
        if err == nil { t.Fatal("should error on non-existent task") }
}

func TestCompleteTaskNotFound(t *testing.T) {
        c := NewCoordinator()
        err := c.CompleteTask("nonexistent", "oracle1")
        if err == nil { t.Fatal("should error on non-existent task") }
}

func TestAbandonTaskNotFound(t *testing.T) {
        c := NewCoordinator()
        err := c.AbandonTask("nonexistent", "oracle1")
        if err == nil { t.Fatal("should error on non-existent task") }
}

func TestAbandonTaskWrongVessel(t *testing.T) {
        c := NewCoordinator()
        c.PostTask(&Task{ID: "t1", Title: "Test"})
        c.ClaimTask("t1", "oracle1")
        err := c.AbandonTask("t1", "jetsonclaw1")
        if err == nil { t.Fatal("should error abandoning someone else's task") }
}

func TestAbandonThenReclaim(t *testing.T) {
        c := NewCoordinator()
        c.PostTask(&Task{ID: "t1", Title: "Test"})
        c.ClaimTask("t1", "oracle1")
        c.AbandonTask("t1", "oracle1")
        // Should be open again, different vessel can claim
        err := c.ClaimTask("t1", "jetsonclaw1")
        if err != nil { t.Fatalf("should allow reclaim after abandon: %v", err) }
        if c.tasks["t1"].ClaimedBy != "jetsonclaw1" { t.Fatal("wrong claimant") }
}

func TestCompleteTaskIncrementsVesselStats(t *testing.T) {
        c := NewCoordinator()
        c.RegisterVessel(VesselInfo{Name: "oracle1"})
        c.PostTask(&Task{ID: "t1", Title: "A"})
        c.PostTask(&Task{ID: "t2", Title: "B"})
        c.ClaimTask("t1", "oracle1")
        c.CompleteTask("t1", "oracle1")
        c.ClaimTask("t2", "oracle1")
        c.CompleteTask("t2", "oracle1")
        if c.vessels["oracle1"].TasksCompleted != 2 {
                t.Fatalf("expected 2 completed, got %d", c.vessels["oracle1"].TasksCompleted)
        }
}

func TestTaskHistoryEmpty(t *testing.T) {
        c := NewCoordinator()
        history := c.TaskHistory("nobody")
        if len(history) != 0 { t.Fatalf("expected empty history, got %d", len(history)) }
}

func TestTaskHistoryOtherVessel(t *testing.T) {
        c := NewCoordinator()
        c.RegisterVessel(VesselInfo{Name: "oracle1"})
        c.PostTask(&Task{ID: "t1", Title: "A"})
        c.ClaimTask("t1", "oracle1")
        c.CompleteTask("t1", "oracle1")
        // Different vessel should see nothing
        history := c.TaskHistory("jetsonclaw1")
        if len(history) != 0 { t.Fatalf("expected empty history for other vessel, got %d", len(history)) }
}

func TestSuggestTasksLimitZero(t *testing.T) {
        c := NewCoordinator()
        c.PostTask(&Task{ID: "t1", Title: "A"})
        c.PostTask(&Task{ID: "t2", Title: "B"})
        // limit 0 means no limit
        suggested := c.SuggestTasks("oracle1", 0)
        if len(suggested) != 2 { t.Fatalf("expected 2, got %d", len(suggested)) }
}

func TestSuggestTasksLimitExceedsAvailable(t *testing.T) {
        c := NewCoordinator()
        c.PostTask(&Task{ID: "t1", Title: "A"})
        suggested := c.SuggestTasks("oracle1", 100)
        if len(suggested) != 1 { t.Fatalf("expected 1, got %d", len(suggested)) }
}

func TestFleetStatusEmpty(t *testing.T) {
        c := NewCoordinator()
        status := c.FleetStatus()
        if status == "" { t.Fatal("status should not be empty even with no data") }
}

func TestFleetStatusContent(t *testing.T) {
        c := NewCoordinator()
        c.RegisterVessel(VesselInfo{Name: "v1"})
        c.RegisterVessel(VesselInfo{Name: "v2"})
        c.PostTask(&Task{ID: "t1", Title: "A"})
        c.PostTask(&Task{ID: "t2", Title: "B"})
        c.ClaimTask("t2", "v1")
        status := c.FleetStatus()
        // Should contain vessel count
        if !contains(status, "2 vessels") { t.Fatalf("expected '2 vessels' in: %s", status) }
        if !contains(status, "1 open") { t.Fatalf("expected '1 open' in: %s", status) }
        if !contains(status, "1 claimed") { t.Fatalf("expected '1 claimed' in: %s", status) }
}

func TestOpenTasksNoVesselRegistered(t *testing.T) {
        c := NewCoordinator()
        c.PostTask(&Task{ID: "t1", Title: "A"})
        c.PostTask(&Task{ID: "t2", Title: "B"})
        open := c.OpenTasks("unknown-vessel")
        if len(open) != 2 { t.Fatalf("expected 2 open, got %d", len(open)) }
}

func TestExportJSONEmpty(t *testing.T) {
        c := NewCoordinator()
        j, err := c.ExportJSON()
        if err != nil { t.Fatal(err) }
        // Empty array (json.MarshalIndent adds trailing newline)
        if j != "[]\n" && j != "[]" {
                t.Fatalf("expected empty JSON array, got: %q", j)
        }
}

func TestPostTaskMultipleTasks(t *testing.T) {
        c := NewCoordinator()
        for i := 0; i < 10; i++ {
                err := c.PostTask(&Task{ID: fmt.Sprintf("task-%d", i), Title: fmt.Sprintf("Task %d", i)})
                if err != nil { t.Fatal(err) }
        }
        open := c.OpenTasks("anyone")
        if len(open) != 10 { t.Fatalf("expected 10 open tasks, got %d", len(open)) }
}

func contains(s, substr string) bool {
        return strings.Contains(s, substr)
}
