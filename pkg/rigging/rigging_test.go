package rigging

import (
        "testing"

        "github.com/SuperInstance/greenhorn-runtime/pkg/allocator"
        "github.com/SuperInstance/greenhorn-runtime/pkg/profiler"
)

// --- AutoSelect tests ---

func TestAutoSelect(t *testing.T) {
        tests := []struct {
                name    string
                profile profiler.Profile
                want    string
        }{
                {
                        name:    "high-end GPU selects compute",
                        profile: profiler.Profile{HasGPU: true, VRAMMB: 8192, RAMMB: 32768},
                        want:    "compute",
                },
                {
                        name:    "exactly 4096 VRAM GPU selects compute",
                        profile: profiler.Profile{HasGPU: true, VRAMMB: 4096, RAMMB: 16384},
                        want:    "compute",
                },
                {
                        name:    "low VRAM GPU falls through to RAM check",
                        profile: profiler.Profile{HasGPU: true, VRAMMB: 2048, RAMMB: 8192},
                        want:    "coder",
                },
                {
                        name:    "high RAM no GPU selects coder",
                        profile: profiler.Profile{HasGPU: false, RAMMB: 8192},
                        want:    "coder",
                },
                {
                        name:    "exactly 4096 RAM no GPU selects coder",
                        profile: profiler.Profile{HasGPU: false, RAMMB: 4096},
                        want:    "coder",
                },
                {
                        name:    "medium RAM selects thinker",
                        profile: profiler.Profile{HasGPU: false, RAMMB: 2048},
                        want:    "thinker",
                },
                {
                        name:    "exactly 1024 RAM selects thinker",
                        profile: profiler.Profile{HasGPU: false, RAMMB: 1024},
                        want:    "thinker",
                },
                {
                        name:    "low RAM selects scavenger",
                        profile: profiler.Profile{HasGPU: false, RAMMB: 512},
                        want:    "scavenger",
                },
                {
                        name:    "minimal RAM selects scavenger",
                        profile: profiler.Profile{HasGPU: false, RAMMB: 256},
                        want:    "scavenger",
                },
                {
                        name:    "zero RAM selects scavenger",
                        profile: profiler.Profile{HasGPU: false, RAMMB: 0},
                        want:    "scavenger",
                },
        }
        for _, tt := range tests {
                t.Run(tt.name, func(t *testing.T) {
                        got := AutoSelect(&tt.profile)
                        if got != tt.want {
                                t.Errorf("AutoSelect() = %q, want %q", got, tt.want)
                        }
                })
        }
}

// --- Deploy tests ---

func TestDeploy(t *testing.T) {
        tests := []struct {
                name    string
                rigging string
                wantErr bool
                wantGPU bool
        }{
                {name: "scout", rigging: "scout", wantErr: false, wantGPU: false},
                {name: "coder", rigging: "coder", wantErr: false, wantGPU: false},
                {name: "compute", rigging: "compute", wantErr: false, wantGPU: true},
                {name: "thinker", rigging: "thinker", wantErr: false, wantGPU: false},
                {name: "scavenger", rigging: "scavenger", wantErr: false, wantGPU: false},
                {name: "unknown", rigging: "foobar", wantErr: true},
        }
        p := profiler.Profile{OS: "linux", Arch: "amd64", CPUCores: 4, RAMMB: 8192}
        res := allocator.AutoFromProfile(&p)
        for _, tt := range tests {
                t.Run(tt.name, func(t *testing.T) {
                        rig, err := Deploy(tt.rigging, p, res)
                        if tt.wantErr {
                                if err == nil {
                                        t.Fatal("expected error")
                                }
                                return
                        }
                        if err != nil {
                                t.Fatalf("unexpected error: %v", err)
                        }
                        if rig.Name != tt.rigging {
                                t.Errorf("Name = %q, want %q", rig.Name, tt.rigging)
                        }
                        if rig.GPU != tt.wantGPU {
                                t.Errorf("GPU = %v, want %v", rig.GPU, tt.wantGPU)
                        }
                        if len(rig.Capabilities) == 0 {
                                t.Error("capabilities should not be empty")
                        }
                })
        }
}

// --- CanCompute tests ---

func TestCanCompute(t *testing.T) {
        tests := []struct {
                name string
                rig  *Rigging
                want bool
        }{
                {name: "GPU rigging", rig: &Rigging{GPU: true, RAMMB: 0}, want: true},
                {name: "high RAM", rig: &Rigging{GPU: false, RAMMB: 2048}, want: true},
                {name: "exactly 512 RAM", rig: &Rigging{GPU: false, RAMMB: 512}, want: true},
                {name: "low RAM", rig: &Rigging{GPU: false, RAMMB: 256}, want: false},
                {name: "zero RAM", rig: &Rigging{GPU: false, RAMMB: 0}, want: false},
        }
        for _, tt := range tests {
                t.Run(tt.name, func(t *testing.T) {
                        if got := tt.rig.CanCompute(); got != tt.want {
                                t.Errorf("CanCompute() = %v, want %v", got, tt.want)
                        }
                })
        }
}

// --- Execute tests ---

func TestExecute(t *testing.T) {
        rig := &Rigging{Name: "test"}
        executed := false
        rig.Execute(func() {
                executed = true
        })
        if !executed {
                t.Fatal("execute function should have been called")
        }
}

func TestExecuteNil(t *testing.T) {
        rig := &Rigging{Name: "test"}
        rig.Execute(nil) // should not panic
}

// --- Park tests ---

func TestPark(t *testing.T) {
        rig := &Rigging{Name: "test", executeFn: func() {}}
        rig.Park()
        if rig.executeFn != nil {
                t.Fatal("executeFn should be nil after Park")
        }
}

// --- Rigging property tests ---

func TestRiggingProperties(t *testing.T) {
        // Verify each rigging type has expected properties
        riggings := []string{"scout", "coder", "compute", "thinker", "scavenger"}
        p := profiler.Profile{OS: "linux", Arch: "amd64", CPUCores: 4, RAMMB: 8192}
        res := allocator.AutoFromProfile(&p)

        for _, name := range riggings {
                t.Run(name, func(t *testing.T) {
                        rig, err := Deploy(name, p, res)
                        if err != nil {
                                t.Fatal(err)
                        }
                        if rig.Name != name {
                                t.Errorf("Name = %q, want %q", rig.Name, name)
                        }
                        if rig.Description == "" {
                                t.Error("Description should not be empty")
                        }
                        if rig.RAMMB <= 0 {
                                t.Errorf("RAMMB = %d, should be > 0", rig.RAMMB)
                        }
                        if rig.APIMode == "" {
                                t.Error("APIMode should not be empty")
                        }
                })
        }
}
