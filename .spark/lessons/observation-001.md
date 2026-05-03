---
room: lessons
type: observation
id: 001
author: greenhorn-runtime
timestamp: 2026-05-03T17:57:00Z
confidence: high
tags: [fleet, cocapn]
references: []
---

# The Cocapn Fleet Runtime Topology

greenhorn-runtime targets the Cocapn Fleet's diverse hardware:
- 🔮 Oracle1: ARM64 Oracle Cloud (Linux, 24GB)
- ⚡ JetsonClaw1: Jetson Orin (ARM64 + CUDA 1024 cores)
- ⚒️ Forgemaster: x86_64 RTX 4050 (Linux)
- CCC: Cloud-hosted (Telegram)

Each hardware target may need a different language backend:
- Go: Oracle1 (ARM64 cloud)
- C/CUDA: JetsonClaw1 (GPU edge)
- Rust: Forgemaster (GPU foundry)
- Zig: Embedded targets
- Python: Universal scripting

The runtime abstracts over these. Agents write once; the runtime deploys anywhere.
