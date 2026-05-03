# Spark Shell — greenhorn-runtime

## Protocol
Version: 1.0 | Storage: `.spark/` directory (git-tracked)

## What is greenhorn-runtime
Portable agent runtime in Go, C, C++, CUDA, Rust, Zig, Java, JS, Python.
Plants agents anywhere within hardware/API limits.
Part of the Cocapn Fleet Bootstrap Bomb/Spark protocol stack.

## Rooms
- **domain/** — what this runtime is (portable multi-language runtime)
- **lessons/** — what happened (runtime deployments, porting experiences)
- **active/** — what's happening now
- **decisions/** — why choices were made (language choices, API decisions)
- **questions/** — what we don't know

## Naming
`[room]-[type]-[id].md`

## Tile Format
```markdown
---
room: [room]
type: [type]
id: [###]
author: greenhorn-runtime
timestamp: [ISO timestamp]
confidence: high|medium|low
tags: [tags]
references: [other-tile-ids]
---
[Content]
```

## Connection to Fleet
- Bootstrap Spark → Bootstrap Bomb → PLATO → greenhorn-runtime
- See: purplepincher.org, github.com/SuperInstance/flux-research

## Manifest
| Room | Tiles | Updated |
|------|--------|---------|
| domain/ | 2 | 2026-05-03 |
| lessons/ | 1 | 2026-05-03 |
| active/ | 1 | 2026-05-03 |
| decisions/ | 1 | 2026-05-03 |
| questions/ | 1 | 2026-05-03 |
