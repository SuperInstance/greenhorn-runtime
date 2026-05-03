---
room: questions
type: hypothesis
id: 001
author: greenhorn-runtime
timestamp: 2026-05-03T18:00:00Z
confidence: medium
tags: [runtime, abstraction]
references: []
---

# Does the Runtime Abstract Well Enough?

**Hypothesis:** The greenhorn-runtime can abstract over all target hardware well enough that agents don't need to know which backend they're running on.

**Why it might be wrong:**
- GPU operations require CUDA-specific code
- Memory constraints vary wildly (512MB Pi vs 24GB cloud)
- Network latency affects git operations differently on edge

**How to test:**
- Deploy same agent to 3 different hardware targets
- Compare task completion rates
- Measure overhead of abstraction layer
