# 🌱 greenhorn-runtime

> *A portable agent runtime that plants anywhere within resource limits.*

The Greenhorn Runtime is the Cocapn Fleet's portable agent deployment target. It's designed to run on constrained hardware — the "greenhorn" gets assigned to whatever hardware is available, from a VPS to a Jetson to a Raspberry Pi.

## What It Does

- **Discovers** fleet repos via GitHub API
- **Clones** the vessel repo for its assigned agent
- **Reads** the taskboard and picks work
- **Executes** tasks using the best available tools
- **Pushes** results back via git
- **Reports** status via message-in-a-bottle

## Architecture

```
greenhorn-runtime/
├── bin/
│   └── greenhorn          # Main entry point
├── lib/
│   ├── discovery.py       # Fleet repo discovery
│   ├── vessel.py          # Vessel cloning and setup
│   ├── taskboard.py       # Task reading and claiming
│   ├── executor.py        # Task execution engine
│   └── reporter.py        # Status reporting via git
├── config/
│   └── default.yaml       # Default configuration
├── Dockerfile             # Container deployment
└── README.md
```

## The Bootstrap Stack

greenhorn-runtime is the **deployment layer** of the Cocapn Fleet's Bootstrap protocol:

```
Bootstrap Spark (universal entry — any project, any domain)
    ↓
Bootstrap Bomb (fleet self-assembly via PLATO)
    ↓
PLATO room server (shared knowledge lattice)
    ↓
greenhorn (dojo model — agent growth framework)
    ↓
greenhorn-runtime (portable deployment)
```

## Fleet Hardware Targets

greenhorn-runtime targets the full Cocapn Fleet:

| Vessel | Hardware | Language Backend |
|--------|----------|-----------------|
| 🔮 Oracle1 | ARM64 Oracle Cloud (24GB) | Go |
| ⚡ JetsonClaw1 | Jetson Orin (CUDA 1024 cores) | C/CUDA |
| ⚒️ Forgemaster | RTX 4050 x86_64 | Rust |
| 🦀 CCC | Cloud-hosted | Python |

## Quick Start

### Local (Python)
```bash
pip install -r requirements.txt
python -m greenhorn --vessel https://github.com/SuperInstance/your-vessel
```

### Docker
```bash
docker build -t greenhorn .
docker run -e GITHUB_TOKEN=$TOKEN greenhorn
```

### Codespace
Open in GitHub Codespaces — pre-configured.

## Resource Limits

- **RAM**: 512MB minimum
- **CPU**: 1 core minimum
- **Disk**: 1GB minimum
- **Network**: GitHub API access required

## Key Papers

| Paper | What It Explains |
|-------|-----------------|
| [*The Bootstrap Spark*](https://github.com/SuperInstance/flux-research/blob/main/whitepapers/2026-05-03-bootstrap-spark.md) | Universal minimum ignition state for any project |
| [*The Bootstrap Bomb*](https://github.com/SuperInstance/flux-research/blob/main/whitepapers/2026-05-03-bootstrap-bomb.md) | Fleet self-assembly through PLATO |
| [*The Dojo Model*](https://github.com/SuperInstance/flux-research/blob/main/research/whitepapers/2026-05-01-dojo-model.md) | Train crew while catching fish |

## Related Repos

- [greenhorn](https://github.com/SuperInstance/greenhorn) — the dojo model for agent growth
- [greenhorn-onboarding](https://github.com/SuperInstance/greenhorn-onboarding) — zero-config fleet onboarding
- [purplepincher.org](https://purplepincher.org) — agent shell technology
- [flux-research](https://github.com/SuperInstance/flux-research) — all fleet papers
- [PLATO room server](https://github.com/SuperInstance/plato-room-phi) — shared knowledge lattice

---

*Built by the Cocapn Fleet. The ocean counts. The Spark lights the fire.*
