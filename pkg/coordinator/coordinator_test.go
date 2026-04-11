package coordinator

import "testing"

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
