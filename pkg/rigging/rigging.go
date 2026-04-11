package rigging

import (
	"fmt"

	"github.com/SuperInstance/greenhorn-runtime/pkg/allocator"
	"github.com/SuperInstance/greenhorn-runtime/pkg/profiler"
)

type Rigging struct {
	Name        string
	Description string
	GPU         bool
	RAMMB       int64
	APIMode     string // readonly, write, full, ideation, minimal
	Capabilities []string
	executeFn   func()
}

var riggings = map[string]func(profiler.Profile, *allocator.ResourceConfig) *Rigging{
	"scout": func(p profiler.Profile, r *allocator.ResourceConfig) *Rigging {
		return &Rigging{
			Name:        "scout",
			Description: "Lightweight repo scanner, no GPU needed",
			GPU:         false,
			RAMMB:       512,
			APIMode:     "readonly",
			Capabilities: []string{"scan", "index", "map", "audit"},
		}
	},
	"coder": func(p profiler.Profile, r *allocator.ResourceConfig) *Rigging {
		return &Rigging{
			Name:        "coder",
			Description: "Code generation and PR creation",
			GPU:         false,
			RAMMB:       2048,
			APIMode:     "write",
			Capabilities: []string{"generate", "refactor", "test", "document"},
		}
	},
	"compute": func(p profiler.Profile, r *allocator.ResourceConfig) *Rigging {
		return &Rigging{
			Name:        "compute",
			Description: "GPU compute for tensors, benchmarks, training",
			GPU:         true,
			RAMMB:       8192,
			APIMode:     "full",
			Capabilities: []string{"tensor", "benchmark", "train", "cuda"},
		}
	},
	"thinker": func(p profiler.Profile, r *allocator.ResourceConfig) *Rigging {
		return &Rigging{
			Name:        "thinker",
			Description: "Think tank ideation, roundtables",
			GPU:         false,
			RAMMB:       1024,
			APIMode:     "ideation",
			Capabilities: []string{"ideate", "roundtable", "brainstorm", "attack"},
		}
	},
	"scavenger": func(p profiler.Profile, r *allocator.ResourceConfig) *Rigging {
		return &Rigging{
			Name:        "scavenger",
			Description: "Beachcomb sweeps, cron jobs, free tier scavenging",
			GPU:         false,
			RAMMB:       256,
			APIMode:     "minimal",
			Capabilities: []string{"sweep", "cron", "scavenge", "monitor"},
		}
	},
}

func AutoSelect(p *profiler.Profile) string {
	if p.HasGPU && p.VRAMMB >= 4096 {
		return "compute"
	}
	if p.RAMMB >= 4096 {
		return "coder"
	}
	if p.RAMMB >= 1024 {
		return "thinker"
	}
	return "scavenger"
}

func Deploy(name string, p profiler.Profile, r *allocator.ResourceConfig) (*Rigging, error) {
	fn, ok := riggings[name]
	if !ok {
		return nil, fmt.Errorf("unknown rigging: %s", name)
	}
	rig := fn(p, r)
	return rig, nil
}

func (r *Rigging) CanCompute() bool {
	if r.GPU {
		return true
	}
	return r.RAMMB >= 512
}

func (r *Rigging) Execute(fn func()) {
	r.executeFn = fn
	if fn != nil {
		fn()
	}
}

func (r *Rigging) Park() {
	// Save current state to repo branch
	fmt.Printf("Parking rigging: %s\n", r.Name)
	r.executeFn = nil
}
