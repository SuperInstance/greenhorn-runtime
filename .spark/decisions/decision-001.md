---
room: decisions
type: rationale
id: 001
author: greenhorn-runtime
timestamp: 2026-05-03T17:59:00Z
confidence: high
tags: [multi-language, runtime]
references: []
---

# Why Multi-Language Runtime

**Decision:** Support Go, C, C++, CUDA, Rust, Zig, Java, JS, Python.

**Rationale:** Different hardware targets have different capabilities:
- Cloud VPS → Go (static binary, no runtime deps)
- GPU edge → C/CUDA (direct hardware access)
- Foundry → Rust (memory safety + GPU)
- Embedded → Zig (tiny binaries)
- Universal → Python (easiest agent scripting)

**Alternative rejected:** Single-language runtime. Would limit deployment targets.

**Tradeoff:** More maintenance burden per language. Worth it for deployment flexibility.
