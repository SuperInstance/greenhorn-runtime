# 🌱 greenhorn-runtime

> *A portable agent runtime that plants anywhere within resource limits.*

The Greenhorn Runtime is the FLUX Fleet's portable agent deployment target. It's designed to run on constrained hardware — the "greenhorn" gets assigned to whatever hardware is available, from a VPS to a Jetson to a Raspberry Pi.

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
Open in GitHub Codespaces — it's pre-configured.

## Resource Limits

Designed to run within:
- **RAM**: 512MB minimum
- **CPU**: 1 core minimum
- **Disk**: 1GB minimum
- **Network**: GitHub API access required

## Part of the FLUX Fleet

- **Onboarding**: https://github.com/SuperInstance/greenhorn-onboarding
- **Dashboard**: https://superinstance.github.io/oracle1-index/
- **Tasks**: https://github.com/SuperInstance/SuperInstance/blob/main/message-in-a-bottle/TASKS.md
