package allocator

import (
	"os"
	"time"

	"github.com/SuperInstance/greenhorn-runtime/pkg/profiler"
	"gopkg.in/yaml.v3"
)

type APIConfig struct {
	Name         string `yaml:"name"`
	Key          string `yaml:"key"`
	RateLimit    int    `yaml:"rate_limit"`     // calls per minute
	FreeTierDay  int    `yaml:"free_tier_daily"` // free calls per day
	BudgetDaily  int    `yaml:"budget_daily"`    // paid calls per day
	Schedule     string `yaml:"schedule"`        // cron schedule hint
}

type GPUConfig struct {
	Available    bool   `yaml:"available"`
	VRAMMB       int64  `yaml:"vram_mb"`
	SharedMemory bool   `yaml:"shared_memory"`
	Schedule     string `yaml:"schedule"`
}

type CPUConfig struct {
	Cores  int   `yaml:"cores"`
	Arch   string `yaml:"arch"`
	RAMMB  int64  `yaml:"ram_mb"`
}

type FleetConfig struct {
	Onboarding    string `yaml:"onboarding"`
	PollInterval  string `yaml:"poll_interval"`
}

type ResourceConfig struct {
	GPU      GPUConfig   `yaml:"gpu"`
	CPU      CPUConfig   `yaml:"cpu"`
	APIs     []APIConfig `yaml:"apis"`
	Fleet    FleetConfig `yaml:"fleet"`
	Rigging  string      `yaml:"rigging"`
}

type Budget struct {
	APICalls map[string]int // name → remaining calls today
	GPUMinutes int          // remaining GPU compute minutes
	LastReset time.Time     // when budgets were last reset
}

func LoadConfig(path string) (*ResourceConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg ResourceConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	// Resolve env vars in API keys
	for i := range cfg.APIs {
		cfg.APIs[i].Key = resolveEnv(cfg.APIs[i].Key)
	}
	return &cfg, nil
}

func AutoFromProfile(p *profiler.Profile) *ResourceConfig {
	cfg := &ResourceConfig{
		CPU: CPUConfig{
			Cores: p.CPUCores,
			Arch:  p.Arch,
			RAMMB: p.RAMMB,
		},
		GPU: GPUConfig{
			Available:    p.HasGPU,
			VRAMMB:       p.VRAMMB,
			SharedMemory: p.HasGPU && p.VRAMMB == 0, // Jetson
		},
		Fleet: FleetConfig{
			Onboarding:   "https://github.com/SuperInstance/greenhorn-onboarding",
			PollInterval: "30m",
		},
		Rigging: "auto",
	}
	return cfg
}

func NewBudget(cfg *ResourceConfig) *Budget {
	b := &Budget{
		APICalls:   make(map[string]int),
		GPUMinutes: 60, // default
		LastReset:  time.Now(),
	}
	for _, api := range cfg.APIs {
		limit := api.BudgetDaily
		if api.FreeTierDay > 0 {
			limit = api.FreeTierDay
		}
		if limit == 0 {
			limit = 1000 // safe default
		}
		b.APICalls[api.Name] = limit
	}
	return b
}

func (b *Budget) CanCall(api string) bool {
	return b.APICalls[api] > 0
}

func (b *Budget) UseCall(api string) {
	if v, ok := b.APICalls[api]; ok && v > 0 {
		b.APICalls[api] = v - 1
	}
}

func (b *Budget) ResetIfNeeded() {
	if time.Since(b.LastReset) > 24*time.Hour {
		b.LastReset = time.Now()
		for k := range b.APICalls {
			b.APICalls[k] = 1000 // reset to default
		}
		b.GPUMinutes = 60
	}
}

func resolveEnv(s string) string {
	if len(s) > 0 && s[0] == '$' {
		return os.Getenv(s[1:])
	}
	return s
}
