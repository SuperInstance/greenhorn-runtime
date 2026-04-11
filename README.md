# Greenhorn Runtime 🟢

**Download. Deploy. Specialize. Repeat.**

A portable agent runtime that plants itself on any hardware and works within whatever limits it finds — GPU, CPU, RAM, VRAM, API calls, rate limits, free tier quotas.

---

## What It Does

Greenhorn Runtime is a **self-contained agent** that:

1. **Discovers its environment** — scans for GPUs, CPUs, RAM, VRAM, API keys, rate limits
2. **Allocates within limits** — never exceeds what's available
3. **Connects to the fleet** — reads the onboarding repo, finds fences, claims work
4. **Specializes as it goes** — the tasks it completes shape what it becomes
5. **Parks and swaps** — done with one rigging? Park the repo. Pull out another. Like swapping heavy machinery.

## Quick Start

```bash
# Download the runtime (Go binary, single file)
go install github.com/SuperInstance/greenhorn-runtime/cmd/greenhorn@latest

# Or build from source
git clone https://github.com/SuperInstance/greenhorn-runtime
cd greenhorn-runtime
make build

# Run it (give it a PAT and point it at the fleet)
greenhorn --token ghp_xxxxx --fleet https://github.com/SuperInstance/greenhorn-onboarding
```

## Language Implementations

| Language | Status | Use Case |
|----------|--------|----------|
| **Go** | ✅ Primary | Cloud VMs, always-on agents, fleet coordination |
| **C** | ✅ Core | Embedded, edge, minimal overhead |
| **C++** | 🔧 Building | GPU compute, high-performance |
| **CUDA C++** | 🔧 Building | NVIDIA GPU direct, tensor ops |
| **Zig** | 📋 Planned | Bare-metal, WASM targets |
| **Rust** | 📋 Planned | Safety-critical, WebAssembly |

## Architecture

```
┌─────────────────────────────────────────────┐
│            GREENHORN RUNTIME                │
│                                             │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Profiler│  │ Allocator│  │ Connector │  │
│  │ (hw/sw) │  │ (budgets)│  │ (fleet)   │  │
│  └────┬────┘  └─────┬────┘  └─────┬─────┘  │
│       │             │              │         │
│  ┌────▼─────────────▼──────────────▼─────┐  │
│  │            Task Engine                │  │
│  │  (claims fences, executes, reports)   │  │
│  └──────────────────┬───────────────────┘  │
│                     │                       │
│  ┌──────────────────▼───────────────────┐  │
│  │            FLUX VM (embedded)         │  │
│  │  (executes bytecode from vocabulary) │  │
│  └──────────────────────────────────────┘  │
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │         Resource Scheduler           │  │
│  │  (cron for free tiers, rate limits)  │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## Resource-Aware Scheduling

Greenhorn doesn't just run tasks. It schedules around bottlenecks:

- **Free tier maximizer**: Uses daily API quotas before reset on a cron
- **GPU budget**: Tracks VRAM, schedules compute-heavy work when available
- **API rate limiter**: Spreads calls across windows, never hits limits
- **Park and swap**: Done with a heavy compute rigging? Park it. Pull out a lightweight scouting rigging. Same agent, different equipment.

```yaml
# Example: greenhorn.yaml — resource profile
resources:
  gpu:
    available: true
    vram_mb: 8192
    shared_memory: true
    schedule: "compute tasks between 02:00-06:00 UTC"
  
  cpu:
    cores: 8
    arch: arm64
    ram_mb: 16384
  
  apis:
    - name: siliconflow
      key: ${SILICONFLOW_KEY}
      rate_limit: 60/min
      free_tier_daily: 1000
      schedule: "ideation before daily reset"
    
    - name: zai
      key: ${ZAI_KEY}
      rate_limit: 30/min
      budget_daily: 500
    
    - name: deepseek
      key: ${DEEPSEEK_KEY}
      rate_limit: 30/min
      budget_daily: 200

fleet:
  onboarding: https://github.com/SuperInstance/greenhorn-onboarding
  poll_interval: 30m
  
rigging: scout  # Current equipment loadout
```

## Riggings (Equipment Loadouts)

A rigging is a set of capabilities the agent is currently wearing. Like parking a bulldozer and pulling out a surveying transit.

| Rigging | GPU | RAM | APIs | Use Case |
|---------|-----|-----|------|----------|
| **scout** | ❌ | 512MB | Read-only | Repo scanning, indexing |
| **coder** | ❌ | 2GB | Write | Code generation, PRs |
| **compute** | ✅ | 8GB+ | Full | Tensor ops, benchmarks |
| **thinker** | ❌ | 1GB | Ideation only | Think tank, roundtables |
| **scavenger** | ❌ | 256MB | Minimal | Beachcomb sweeps, cron |

Swap riggings without restarting:
```bash
greenhorn rigging park coder      # Park current coder rigging
greenhorn rigging deploy compute  # Pull out compute rigging
greenhorn rigging status          # What's equipped?
```

Each rigging persists as a repo branch: `rigging/coder`, `rigging/compute`, etc.

## Fleet Integration

Greenhorn agents are full fleet members:
- Read the [Fence Board](https://github.com/SuperInstance/oracle1-vessel/blob/main/FENCE-BOARD.md)
- Claim fences, earn badges, grow career stages
- Drop bottles for async communication
- Report status via commit feed

## License

MIT. Plant agents everywhere.

---

*Part of the [Cocapn Fleet](https://github.com/SuperInstance/greenhorn)*
*Greenhorn — because everyone deserves a shot at boat ownership.*
