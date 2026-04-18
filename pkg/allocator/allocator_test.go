package allocator

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/SuperInstance/greenhorn-runtime/pkg/profiler"
)

// --- Budget tests ---

func TestNewBudget(t *testing.T) {
	tests := []struct {
		name       string
		cfg        *ResourceConfig
		wantGPU    int
		wantCalls  int
	}{
		{
			name: "empty config gets defaults",
			cfg:  &ResourceConfig{},
			wantGPU:   60,
			wantCalls: 0,
		},
		{
			name: "free tier takes precedence",
			cfg: &ResourceConfig{
				APIs: []APIConfig{
					{Name: "openai", FreeTierDay: 500, BudgetDaily: 10000},
				},
			},
			wantGPU:   60,
			wantCalls: 500,
		},
		{
			name: "budget daily when no free tier",
			cfg: &ResourceConfig{
				APIs: []APIConfig{
					{Name: "paid-api", BudgetDaily: 2500},
				},
			},
			wantGPU:   60,
			wantCalls: 2500,
		},
		{
			name: "zero limit falls back to 1000",
			cfg: &ResourceConfig{
				APIs: []APIConfig{
					{Name: "unlimited-api", FreeTierDay: 0, BudgetDaily: 0},
				},
			},
			wantGPU:   60,
			wantCalls: 1000,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := NewBudget(tt.cfg)
			if b.GPUMinutes != tt.wantGPU {
				t.Errorf("GPUMinutes = %d, want %d", b.GPUMinutes, tt.wantGPU)
			}
			if len(b.APICalls) > 0 && b.APICalls[tt.cfg.APIs[0].Name] != tt.wantCalls {
				t.Errorf("APICalls[%s] = %d, want %d", tt.cfg.APIs[0].Name, b.APICalls[tt.cfg.APIs[0].Name], tt.wantCalls)
			}
		})
	}
}

func TestBudgetCanCall(t *testing.T) {
	b := NewBudget(&ResourceConfig{
		APIs: []APIConfig{{Name: "test-api", FreeTierDay: 5}},
	})
	if !b.CanCall("test-api") {
		t.Fatal("should be able to call with budget remaining")
	}
	if b.CanCall("unknown-api") {
		t.Fatal("unknown API should not be callable")
	}
	// Drain budget
	for i := 0; i < 5; i++ {
		b.UseCall("test-api")
	}
	if b.CanCall("test-api") {
		t.Fatal("should not be callable when exhausted")
	}
}

func TestBudgetUseCall(t *testing.T) {
	b := NewBudget(&ResourceConfig{
		APIs: []APIConfig{{Name: "test-api", FreeTierDay: 3}},
	})
	b.UseCall("test-api")
	b.UseCall("test-api")
	b.UseCall("test-api")
	// Over-use should not go negative
	b.UseCall("test-api")
	if b.APICalls["test-api"] < 0 {
		t.Fatal("calls should not go below zero")
	}
	// UseCall on unknown API should not panic
	b.UseCall("nonexistent")
}

func TestBudgetResetIfNeeded(t *testing.T) {
	b := NewBudget(&ResourceConfig{
		APIs: []APIConfig{{Name: "api1", FreeTierDay: 10}},
	})
	// Drain
	for i := 0; i < 10; i++ {
		b.UseCall("api1")
	}
	if b.APICalls["api1"] != 0 {
		t.Fatalf("expected 0, got %d", b.APICalls["api1"])
	}
	// Reset not yet triggered
	b.ResetIfNeeded()
	if b.APICalls["api1"] != 0 {
		t.Fatal("should still be 0, not time to reset")
	}
	// Simulate passage of time
	b.LastReset = b.LastReset.Add(-25 * time.Hour)
	b.ResetIfNeeded()
	if b.APICalls["api1"] != 1000 {
		t.Fatalf("after reset expected 1000, got %d", b.APICalls["api1"])
	}
	if b.GPUMinutes != 60 {
		t.Fatalf("GPU minutes should reset to 60, got %d", b.GPUMinutes)
	}
}

// --- AutoFromProfile tests ---

func TestAutoFromProfile(t *testing.T) {
	tests := []struct {
		name      string
		profile   profiler.Profile
		wantRig   string
		wantGPU   bool
	}{
		{
			name: "high-end GPU rigging",
			profile: profiler.Profile{
				HasGPU: true, VRAMMB: 8192, CPUCores: 8, RAMMB: 32768,
			},
			wantRig: "auto",
			wantGPU: true,
		},
		{
			name: "jetson shared memory",
			profile: profiler.Profile{
				HasGPU: true, VRAMMB: 0, CPUCores: 4, RAMMB: 4096,
			},
			wantRig: "auto",
			wantGPU: true, // HasGPU is true
		},
		{
			name: "CPU-only high RAM",
			profile: profiler.Profile{
				HasGPU: false, CPUCores: 4, RAMMB: 8192,
			},
			wantRig: "auto",
			wantGPU: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := AutoFromProfile(&tt.profile)
			if cfg.Rigging != tt.wantRig {
				t.Errorf("Rigging = %q, want %q", cfg.Rigging, tt.wantRig)
			}
			if cfg.GPU.Available != tt.wantGPU {
				t.Errorf("GPU.Available = %v, want %v", cfg.GPU.Available, tt.wantGPU)
			}
			if cfg.CPU.Cores != tt.profile.CPUCores {
				t.Errorf("CPU.Cores = %d, want %d", cfg.CPU.Cores, tt.profile.CPUCores)
			}
			if cfg.Fleet.PollInterval != "30m" {
				t.Errorf("PollInterval = %q, want 30m", cfg.Fleet.PollInterval)
			}
		})
	}
}

// --- LoadConfig tests ---

func TestLoadConfig(t *testing.T) {
	yamlContent := `
gpu:
  available: true
  vram_mb: 4096
cpu:
  cores: 4
  arch: amd64
  ram_mb: 8192
apis:
  - name: openai
    key: $OPENAI_KEY
    rate_limit: 60
    free_tier_daily: 500
fleet:
  onboarding: https://example.com
  poll_interval: 15m
rigging: compute
`
	tmpDir := t.TempDir()
	cfgPath := filepath.Join(tmpDir, "greenhorn.yaml")
	if err := os.WriteFile(cfgPath, []byte(yamlContent), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := LoadConfig(cfgPath)
	if err != nil {
		t.Fatalf("LoadConfig failed: %v", err)
	}
	if cfg.GPU.Available != true {
		t.Fatal("expected GPU available")
	}
	if cfg.GPU.VRAMMB != 4096 {
		t.Fatalf("VRAM = %d, want 4096", cfg.GPU.VRAMMB)
	}
	if cfg.CPU.Cores != 4 {
		t.Fatalf("Cores = %d, want 4", cfg.CPU.Cores)
	}
	if len(cfg.APIs) != 1 || cfg.APIs[0].Name != "openai" {
		t.Fatal("expected 1 API named openai")
	}
	if cfg.Rigging != "compute" {
		t.Fatalf("Rigging = %q, want compute", cfg.Rigging)
	}
}

func TestLoadConfigEnvResolution(t *testing.T) {
	os.Setenv("TEST_SECRET", "resolved-value")
	defer os.Unsetenv("TEST_SECRET")

	yamlContent := `
apis:
  - name: test
    key: $TEST_SECRET
    rate_limit: 10
`
	tmpDir := t.TempDir()
	cfgPath := filepath.Join(tmpDir, "test.yaml")
	os.WriteFile(cfgPath, []byte(yamlContent), 0644)

	cfg, err := LoadConfig(cfgPath)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.APIs[0].Key != "resolved-value" {
		t.Fatalf("key = %q, want resolved-value", cfg.APIs[0].Key)
	}
}

func TestLoadConfigMissingFile(t *testing.T) {
	_, err := LoadConfig("/nonexistent/path.yaml")
	if err == nil {
		t.Fatal("should error on missing file")
	}
}

func TestLoadConfigInvalidYAML(t *testing.T) {
	tmpDir := t.TempDir()
	cfgPath := filepath.Join(tmpDir, "bad.yaml")
	os.WriteFile(cfgPath, []byte("{{invalid yaml"), 0644)

	_, err := LoadConfig(cfgPath)
	if err == nil {
		t.Fatal("should error on invalid YAML")
	}
}

func TestLoadConfigNoEnvPrefix(t *testing.T) {
	yamlContent := `
apis:
  - name: plain
    key: literal_key_value
    rate_limit: 10
`
	tmpDir := t.TempDir()
	cfgPath := filepath.Join(tmpDir, "plain.yaml")
	os.WriteFile(cfgPath, []byte(yamlContent), 0644)

	cfg, err := LoadConfig(cfgPath)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.APIs[0].Key != "literal_key_value" {
		t.Fatalf("key = %q, want literal_key_value", cfg.APIs[0].Key)
	}
}
