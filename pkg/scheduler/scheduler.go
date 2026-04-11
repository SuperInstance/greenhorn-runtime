package scheduler

import (
	"fmt"
	"log"
	"time"

	"github.com/SuperInstance/greenhorn-runtime/pkg/allocator"
	"github.com/SuperInstance/greenhorn-runtime/pkg/connector"
	"github.com/SuperInstance/greenhorn-runtime/pkg/rigging"
)

type Scheduler struct {
	rig       *rigging.Rigging
	resources *allocator.ResourceConfig
	budget    *allocator.Budget
	conn      *connector.Connector
	stop      chan struct{}
}

func New(rig *rigging.Rigging, res *allocator.ResourceConfig, conn *connector.Connector) *Scheduler {
	return &Scheduler{
		rig:       rig,
		resources: res,
		budget:    allocator.NewBudget(res),
		conn:      conn,
		stop:      make(chan struct{}),
	}
}

func (s *Scheduler) Start() {
	log.Println("Scheduler started")

	// Main loop: poll fleet, claim work, execute within budget
	ticker := time.NewTicker(30 * time.Minute)
	defer ticker.Stop()

	// Immediate first poll
	s.pollAndWork()

	for {
		select {
		case <-ticker.C:
			s.pollAndWork()
		case <-s.stop:
			return
		}
	}
}

func (s *Scheduler) Stop() {
	close(s.stop)
}

func (s *Scheduler) pollAndWork() {
	s.budget.ResetIfNeeded()

	// 1. Check for new fences
	fences, err := s.conn.FetchFenceBoard()
	if err != nil {
		log.Printf("Failed to fetch fence board: %v", err)
		return
	}

	// 2. Find fences within our capability
	for _, fence := range fences {
		if fence.Status != "open" {
			continue
		}
		diff, ok := fence.Difficulty[s.rig.Name]
		if !ok || diff > 7 { // don't claim things we'd struggle with
			continue
		}
		if s.canExecute(fence) {
			log.Printf("Claiming fence: %s (difficulty %d)", fence.ID, diff)
			s.execute(fence)
			break // one fence at a time
		}
	}

	// 3. Use free tier quotas before reset (scavenger mode)
	s.scavengeFreeTiers()
}

func (s *Scheduler) canExecute(f connector.Fence) bool {
	// Check if we have the resources for this fence
	return s.rig.CanCompute() && s.budget.GPUMinutes > 0
}

func (s *Scheduler) execute(f connector.Fence) {
	// Claim the fence
	approach := fmt.Sprintf("Auto-claimed by Greenhorn runtime (rigging: %s, profile: %s)",
		s.rig.Name, s.rig.Description)
	if err := s.conn.ClaimFence(f.ID, approach); err != nil {
		log.Printf("Failed to claim fence: %v", err)
		return
	}

	// Execute within budget
	s.rig.Execute(func() {
		log.Printf("Executing fence %s...", f.ID)
		// Actual task execution happens here
		// The rigging determines HOW (FLUX VM, native code, API calls)
	})
}

func (s *Scheduler) scavengeFreeTiers() {
	// Before daily reset, use remaining free tier calls
	for _, api := range s.resources.APIs {
		if api.FreeTierDay > 0 && s.budget.CanCall(api.Name) {
			remaining := s.budget.APICalls[api.Name]
			if remaining > 100 { // worth scavenging
				log.Printf("Scavenging %s: %d free calls remaining", api.Name, remaining)
				// Use remaining calls for ideation/asset generation
				s.useForIdeation(api.Name, remaining/10) // use 10% per poll
			}
		}
	}
}

func (s *Scheduler) useForIdeation(api string, count int) {
	for i := 0; i < count && s.budget.CanCall(api); i++ {
		s.budget.UseCall(api)
		// Make API call for ideation/roundtable/brainstorm
	}
	log.Printf("Used %d calls on %s for ideation", count, api)
}
