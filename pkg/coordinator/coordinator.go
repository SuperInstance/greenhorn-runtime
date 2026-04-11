package coordinator

import (
	"encoding/json"
	"fmt"
	"sort"
	"sync"
	"time"
)

// Task represents a fleet task (from fence board)
type Task struct {
	ID          string   `json:"id"`
	Title       string   `json:"title"`
	PostedBy    string   `json:"posted_by"`
	PostedAt    string   `json:"posted_at"`
	Status      string   `json:"status"` // open, claimed, completed, abandoned
	ClaimedBy   string   `json:"claimed_by,omitempty"`
	Difficulty  map[string]int `json:"difficulty"` // vessel -> difficulty rating
	ClaimWindow string   `json:"claim_window"`
	Reward      string   `json:"reward"`
	Tags        []string `json:"tags"`
}

// Coordinator manages fleet-wide task distribution
type Coordinator struct {
	mu    sync.Mutex
	tasks map[string]*Task
	vessels map[string]VesselInfo
}

type VesselInfo struct {
	Name         string
	Capabilities []string
	Status       string
	LastSeen     time.Time
	TasksCompleted int
}

// NewCoordinator creates a fleet coordinator
func NewCoordinator() *Coordinator {
	return &Coordinator{
		tasks:   make(map[string]*Task),
		vessels: make(map[string]VesselInfo),
	}
}

// PostTask creates a new task on the fence board
func (c *Coordinator) PostTask(task *Task) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if task.ID == "" {
		return fmt.Errorf("task ID required")
	}
	task.Status = "open"
	task.PostedAt = time.Now().UTC().Format(time.RFC3339)
	c.tasks[task.ID] = task
	return nil
}

// ClaimTask assigns a task to a vessel
func (c *Coordinator) ClaimTask(taskID, vessel string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	task, ok := c.tasks[taskID]
	if !ok {
		return fmt.Errorf("task %s not found", taskID)
	}
	if task.Status != "open" {
		return fmt.Errorf("task %s is %s, not open", taskID, task.Status)
	}
	
	// Tom Sawyer rule: the best agent is the one who volunteers
	task.Status = "claimed"
	task.ClaimedBy = vessel
	return nil
}

// CompleteTask marks a task as done
func (c *Coordinator) CompleteTask(taskID, vessel string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	task, ok := c.tasks[taskID]
	if !ok {
		return fmt.Errorf("task %s not found", taskID)
	}
	if task.ClaimedBy != vessel {
		return fmt.Errorf("task %s claimed by %s, not %s", taskID, task.ClaimedBy, vessel)
	}
	
	task.Status = "completed"
	
	// Track vessel stats
	info := c.vessels[vessel]
	info.TasksCompleted++
	c.vessels[vessel] = info
	
	return nil
}

// AbandonTask releases a claimed task
func (c *Coordinator) AbandonTask(taskID, vessel string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	task, ok := c.tasks[taskID]
	if !ok {
		return fmt.Errorf("task %s not found", taskID)
	}
	if task.ClaimedBy != vessel {
		return fmt.Errorf("not your task")
	}
	
	task.Status = "open"
	task.ClaimedBy = ""
	return nil
}

// RegisterVessel adds a vessel to the fleet registry
func (c *Coordinator) RegisterVessel(info VesselInfo) {
	c.mu.Lock()
	defer c.mu.Unlock()
	info.LastSeen = time.Now()
	c.vessels[info.Name] = info
}

// OpenTasks returns all unclaimed tasks, sorted by best fit for a vessel
func (c *Coordinator) OpenTasks(vessel string) []*Task {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	var open []*Task
	for _, t := range c.tasks {
		if t.Status == "open" {
			open = append(open, t)
		}
	}
	
	// Sort by difficulty for this vessel (easiest first = Tom Sawyer encouragement)
	vi, hasVessel := c.vessels[vessel]
	_ = vi
	_ = hasVessel
	
	sort.Slice(open, func(i, j int) bool {
		di, oki := open[i].Difficulty[vessel]
		dj, okj := open[j].Difficulty[vessel]
		if oki && okj {
			return di < dj // easiest first
		}
		if oki { return true }
		return false
	})
	
	return open
}

// FleetStatus returns a summary of fleet state
func (c *Coordinator) FleetStatus() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	open, claimed, completed := 0, 0, 0
	for _, t := range c.tasks {
		switch t.Status {
		case "open": open++
		case "claimed": claimed++
		case "completed": completed++
		}
	}
	
	return fmt.Sprintf("Fleet: %d vessels | Tasks: %d open, %d claimed, %d completed",
		len(c.vessels), open, claimed, completed)
}

// TaskHistory returns completed tasks for a vessel
func (c *Coordinator) TaskHistory(vessel string) []*Task {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	var history []*Task
	for _, t := range c.tasks {
		if t.Status == "completed" && t.ClaimedBy == vessel {
			history = append(history, t)
		}
	}
	return history
}

// SuggestTasks recommends tasks for a vessel based on capabilities and history
func (c *Coordinator) SuggestTasks(vessel string, limit int) []*Task {
	open := c.OpenTasks(vessel)
	if limit > 0 && len(open) > limit {
		open = open[:limit]
	}
	return open
}

// ExportJSON exports all tasks as JSON
func (c *Coordinator) ExportJSON() (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	tasks := make([]*Task, 0, len(c.tasks))
	for _, t := range c.tasks {
		tasks = append(tasks, t)
	}
	
	data, err := json.MarshalIndent(tasks, "", "  ")
	return string(data), err
}
