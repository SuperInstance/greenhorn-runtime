# FLUX CUDA VM

Parallel bytecode execution on NVIDIA GPU. Designed for Jetson Super Orin Nano.

Each CUDA thread runs one FLUX program independently. Batch 1000+ programs
onto 1024 CUDA cores for massive parallel execution.

## Build (requires CUDA toolkit)
```bash
make
```

## Run
```bash
./flux_cuda
```

## Architecture
- **Kernel**: `flux_batch_kernel` — one thread per FLUX program
- **Memory**: N × FLUX_MAX_BC bytecodes, N results, N cycle counts
- **Safety**: 10,000 instruction limit per program to prevent infinite loops
- **Registers**: 16 GP registers per thread (GPU shared memory constraint)

## Benchmark
Expected on Jetson Orin Nano:
- 1000 factorial programs in <5ms
- ~200,000 programs/second throughput
- Serial C vs parallel CUDA comparison

Fence-0x48 — JetsonClaw1 to benchmark on real hardware.
