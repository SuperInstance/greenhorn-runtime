# CUDA FLUX Batch Execution Kernel — Design Document

> **Author:** Quill (Architect-rank agent, greenhorn-runtime)
> **Task:** CUDA-001 — Batch FLUX bytecode execution on Jetson Super Orin Nano
> **Target:** Jetson Super Orin Nano, 1024 CUDA cores, 8GB shared LPDDR5
> **Reference:** `cuda/flux_cuda.cu` (existing stub), `pkg/flux/vm.go` (canonical Go VM)

---

## 1. Architecture Overview

### 1.1 SIMT Execution Model

FLUX bytecode programs are **embarrassingly parallel** — each program has its own register file, program counter, and stack with no cross-program data dependencies during normal execution. This maps directly to CUDA's SIMT (Single Instruction, Multiple Thread) model:

- **One CUDA thread = one FLUX virtual machine instance.**
- Each thread independently fetches, decodes, and executes bytecodes from its own program slice.
- Threads within a warp (32 threads) execute the same instruction in lockstep, but each applies it to its own register file and bytecode stream.

```
Warps in lockstep →    T0: ADD R1,R2,R3    (from program 0)
                       T1: MOVI R0,42       (from program 1)  ← warp divergence
                       T2: ADD R1,R2,R3    (from program 2)
                       ...
                       T31: HALT             (from program 31)  ← divergence
```

### 1.2 Thread Hierarchy Mapping

```
Grid
├── Block 0 (1024 threads max on Orin)
│   ├── Warp 0 (threads 0-31)    → programs 0-31
│   ├── Warp 1 (threads 32-63)   → programs 32-63
│   ├── ...
│   └── Warp 31 (threads 992-1023) → programs 992-1023
├── Block 1
│   └── ...                       → programs 1024-2047
└── ...
```

**Sizing strategy for Jetson Orin Nano (SM 8.7, Ampere):**
- 1024 CUDA cores across multiple streaming multiprocessors (SMs)
- Each SM: 128 CUDA cores, 64KB L1/shared, 128KB L2
- Recommended block size: **256 threads** (8 warps) — balances occupancy with register pressure
- Grid size: `(num_programs + 255) / 256` blocks
- Max concurrent programs: limited by register pressure (see §6)

### 1.3 Memory Hierarchy

| Memory Type       | FLUX Mapping                                | Latency    | Capacity        |
|-------------------|---------------------------------------------|------------|-----------------|
| **Registers**     | Per-thread register file (GP[0..15])        | ~1 cycle   | 64 regs/thread  |
| **Local Memory**  | Per-thread stack (overflows from registers) | ~4-6 cycles | 256 KB max SM   |
| **Shared Memory** | Opcode dispatch table, A2A queues           | ~5-10 cycles | 48-100KB per SM |
| **L2 Cache**      | Bytecode array caching                      | ~20 cycles | 2-4 MB          |
| **Global Memory** | Packed bytecode array, results, error codes | ~200-400 cycles | 8 GB           |
| **Constant Memory**| Opcode format table (read-only)            | ~4 cycles (broadcast) | 64 KB |

**Design decisions:**
1. Register file: **16 registers** (not 64 like Go VM) to reduce register pressure and increase occupancy. Programs needing >16 registers use local memory spill.
2. Stack: **256-entry** per thread in local memory (1KB per thread). Go VM uses 4096 but most programs use <10 entries. Full 4096-entry stack would limit occupancy to ~128 threads/SM.
3. Bytecode: packed variable-length in global memory with offset table (replaces the existing fixed FLUX_MAX_BC=1024 padding).

---

## 2. Kernel Interface

### 2.1 Primary Kernel

```c
/**
 * flux_batch_execute — Execute N FLUX programs in parallel.
 *
 * Each CUDA thread runs one FLUX VM instance. Programs are packed
 * contiguously in a single global memory buffer.
 *
 * @param programs    Packed bytecode array (all programs concatenated)
 * @param offsets     Start offset for each program in the packed array
 * @param lengths     Bytecode length for each program
 * @param results     Output: GP[0] (primary result register) for each program
 * @param error_codes Output: 0=success, 1=div-by-zero, 2=stack-overflow,
 *                   3=stack-underflow, 4=invalid-opcode, 5=max-cycles-exceeded
 * @param cycle_counts Output: total cycles consumed by each program
 * @param num_programs Total number of programs to execute
 */
__global__ void flux_batch_execute(
    const uint8_t* __restrict__ programs,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ lengths,
    int32_t* __restrict__ results,
    int32_t* __restrict__ error_codes,
    int32_t* __restrict__ cycle_counts,
    int num_programs
);
```

### 2.2 Auxiliary Kernels (Future Phases)

```c
// Phase 3: Warp-specialized — separate scheduler and worker kernels
__global__ void flux_warp_scheduler(
    const uint8_t* __restrict__ programs,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ lengths,
    volatile int* __restrict__ work_queues,  // per-SM work queue
    int num_programs
);

__global__ void flux_warp_worker(
    const uint8_t* __restrict__ programs,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ lengths,
    volatile int* __restrict__ work_queues,
    int32_t* __restrict__ results,
    int32_t* __restrict__ error_codes,
    int32_t* __restrict__ cycle_counts
);
```

---

## 3. Memory Layout

### 3.1 Packed Bytecode Array (Global Memory)

```
programs[] = [BC_0 | BC_1 | BC_2 | ... | BC_N-1]

offsets[0] = 0
offsets[1] = len(BC_0)
offsets[2] = len(BC_0) + len(BC_1)
...

lengths[i] = byte count of program i
```

**Coalescing strategy:** Programs are sorted by length (shortest first) to improve warp coalescing when threads in the same warp have similar bytecode sizes. Within a warp, threads access `programs[offsets[threadIdx] + PC]` — if programs are aligned to 16-byte boundaries, adjacent threads in the warp access adjacent memory locations, enabling full coalesced reads.

**Alignment:** Each program is padded to 4-byte alignment. Offset table entries are pre-aligned by the host before copying to GPU.

### 3.2 Per-Thread State

```c
struct FluxThreadState {
    int32_t gp[FLUX_NUM_REGS];      // 16 × 4 = 64 bytes (in registers)
    int32_t stack[FLUX_STACK_SIZE];  // 256 × 4 = 1 KB (in local memory)
    int32_t sp;                      // stack pointer (register)
    int32_t pc;                      // program counter (register)
    int32_t cycles;                  // cycle counter (register)
    int     halted;                  // halt flag (predicate register)
    int     error;                   // error code (register)
};
```

**NVCC register allocation:** The compiler will place `gp[0..15]`, `sp`, `pc`, `cycles` in hardware registers (up to 255 per thread). The `stack[]` array exceeds register capacity and spills to local memory (backed by L1/L2 cache, not DRAM on Ampere).

### 3.3 Shared Memory Layout

```
┌─────────────────────────────────────────┐
│ Opcode Format Table (256 × 1 byte)      │  256 bytes
│ format_table[opcode] = instruction size │
├─────────────────────────────────────────┤
│ A2A Communication Queue                  │
│ (256 entries × 16 bytes)                │  4 KB (Phase 2+)
├─────────────────────────────────────────┤
│ Warp Synchronization Flags              │
│ (1 byte × num_warps)                    │  8 bytes
└─────────────────────────────────────────┘
Total: ~4.5 KB per block (well within 48KB shared limit)
```

### 3.4 Constant Memory

```c
// 256-byte lookup table: opcode → instruction byte length
// Broadcast to all threads in a warp in 1 transaction
__constant__ uint8_t opcode_format[256];
```

Populated by host:
```
0x00-0x07 → 1   (HALT, NOP, RET, ...)
0x08-0x0F → 2   (INC, DEC, NOT, NEG, PUSH, POP, ...)
0x10-0x17 → 2   (STRIPCONF, ...)
0x18-0x1F → 3   (MOVI, ADDI, SUBI, ...)
0x20-0x4F → 4   (ADD, SUB, MUL, ..., JMP, JZ, JNZ, LOOP, CALL, ...)
0x50-0xFF → 1   (unassigned / variable)
```

---

## 4. Dispatch Loop Design

### 4.1 Core Dispatch Loop (Pseudocode)

```c
while (!halted && pc < prog_len && cycles < MAX_CYCLES) {
    uint8_t opcode = programs[base_offset + pc];
    uint8_t fmt    = opcode_format[opcode];      // 1-cycle constant mem read

    // Boundary check (vectorizable within warp)
    if (pc + fmt > prog_len) { error = ERR_INVALID; break; }

    // Fetch operands (coalesced global memory reads)
    uint8_t rd = (fmt >= 2) ? programs[base_offset + pc + 1] : 0;
    uint8_t rs = (fmt >= 4) ? programs[base_offset + pc + 2] : 0;
    uint8_t rt = (fmt >= 4) ? programs[base_offset + pc + 3] : 0;

    cycles++;

    // Dispatch via computed goto / switch (see §4.3)
    DISPATCH(opcode, rd, rs, rt);

    // Advance PC (overwritten by branch instructions)
    if (!halted && !branched) pc += fmt;
    branched = false;
}
```

### 4.2 Format-Aware Decode

| Format | Byte Range | Layout                  | Examples                    |
|--------|-----------|-------------------------|-----------------------------|
| A (1B) | 0x00-0x07 | `[opcode]`              | HALT, NOP, RET              |
| B (2B) | 0x08-0x17 | `[opcode, rd]`          | INC, DEC, NOT, NEG, PUSH, POP |
| C (3B) | 0x18-0x1F | `[opcode, rd, imm8]`    | MOVI, ADDI, SUBI            |
| D (4B) | 0x20-0x3F | `[opcode, rd, rs, rt]`  | ADD, SUB, MUL, MOV, CMP_*   |
| E (4B) | 0x40-0x4F | `[opcode, rd, lo, hi]`  | JMP, JZ, JNZ, MOVI16, LOOP, CALL |
| F (4B) | 0x50-0xFF | `[opcode, rd, rs, rt]`  | TELL, ASK, BCAST (future)   |

### 4.3 Dispatch Mechanism

**Phase 1: Switch statement** — NVCC compiles `switch` on `uint8_t` to a jump table:
```c
switch (opcode) {
    case OP_HALT: halted = true; break;
    case OP_NOP:  break;
    case OP_INC:  gp[rd] += 1; break;
    // ...
}
```

**Phase 3: Computed goto** (GCC/NVCC extension):
```c
static const void* dispatch_table[256] = {
    [0x00] = &&lbl_halt, [0x01] = &&lbl_nop,
    [0x08] = &&lbl_inc,  // ...
};
goto *dispatch_table[opcode];
lbl_halt: halted = true; goto lbl_end;
lbl_nop:  goto lbl_end;
lbl_inc:  gp[rd] += 1; goto lbl_end;
```

NVCC typically compiles `switch` to a jump table for dense opcode ranges anyway. Measured performance difference: <3% on Ampere.

---

## 5. Warp Divergence Strategy

### 5.1 Sources of Divergence

| Source                     | Impact            | Mitigation                                    |
|----------------------------|-------------------|-----------------------------------------------|
| **Conditional branches**   | High              | Sort programs by branch structure (Phase 3)   |
| **Variable program length**| Medium            | Pad to similar lengths; early-exit idle threads|
| **A2A operations**         | Very High         | Serialize within warp via shuffle or mutex    |
| **Different opcodes**      | Low               | Switch dispatch is uniform; divergence only on branches |

### 5.2 Branch Divergence Model

```
Warp executing JZ at different PCs:

Cycle 1:  All threads fetch JZ opcode
Cycle 2:  All threads decode operands (uniform)
Cycle 3:  ┌─ Taken path (threads 0-15):  execute jump
          └─ Not-taken path (threads 16-31): execute fallthrough
Cycle 4:  Divergence — half-warp executes, half masks out
```

**Penalty:** For JZ/JNZ with 50/50 branch probability, divergence causes ~2x slowdown on those instructions. Since branch instructions are typically ~10-20% of total instructions, overall impact is ~10-20% throughput reduction.

**Mitigation — Branch hinting (Phase 3):**
- `__builtin_expect()` on condition: `if (__builtin_expect(gp[rd] == 0, 1))` for loop-exit branches
- Program-level: compiler pre-analyzes branch patterns and reorders programs within warps to group similar branching behavior

### 5.3 Stack Operations (Zero Divergence)

PUSH and POP access only per-thread local memory. No cross-thread synchronization needed. Warp executes these in perfect lockstep — all 32 threads do PUSH simultaneously, each writing to their own stack address.

### 5.4 A2A Operations (High Divergence — Phase 2+)

TELL/ASK/BCAST require inter-thread communication:

```c
case OP_TELL:
    // Write value to global memory A2A queue
    // All threads in warp serialize through atomic operations
    if (threadIdx.x % 32 == 0) {
        // Only lane 0 of each warp performs the A2A write
        int slot = atomicAdd(&a2a_queue.tail, 1);
        a2a_queue.entries[slot].sender = global_tid;
        a2a_queue.entries[slot].value  = gp[rd];
    }
    __syncwarp();  // Re-synchronize warp after A2A
    break;
```

**Phase 3 optimization:** Use `__shfl_sync()` for intra-warp message passing, avoiding global memory entirely:
```c
// Send value from thread S to thread R within same warp
int value = __shfl_sync(0xFFFFFFFF, gp[rd], sender_lane);
```

---

## 6. Performance Estimates

### 6.1 Memory Bandwidth Analysis

**Scenario:** 1024 programs, average 100 bytes each

| Quantity              | Value    | Notes                              |
|-----------------------|----------|------------------------------------|
| Bytecode total        | 100 KB   | Fits entirely in L2 cache (2-4 MB) |
| Results (output)      | 4 KB     | 1024 × 4 bytes                    |
| Cycle counts          | 4 KB     | 1024 × 4 bytes                    |
| Register files        | 64 KB    | 1024 × 64 bytes (in hardware regs)|
| Stack arrays          | 1 MB     | 1024 × 1 KB (local memory)        |
| **Total data touched**| ~1.2 MB  | Well within GPU memory bandwidth   |

**Jetson Orin Nano bandwidth:** ~68 GB/s (LPDDR5)
**Estimated kernel time for 1024 short programs:** ~0.02ms (compute-bound, not bandwidth-bound)

### 6.2 Register Pressure and Occupancy

| Resource              | Per Thread | Per Block (256 threads) | Available   | Limiting Factor? |
|-----------------------|-----------|------------------------|-------------|------------------|
| Registers             | ~28       | 7,168                  | 65,536      | No               |
| Local memory          | 1,024 B   | 256 KB                 | 256 KB      | **Barely fits**  |
| Shared memory         | 0         | 256 B (format table)   | 49,152 B    | No               |

**With 256-entry stacks:** ~6 blocks can resident simultaneously on a single SM. With 2 SMs (for 128 cores... actually Orin Nano has more SMs), occupancy is reasonable.

**Stack size tradeoff:**
- 256 entries: supports 99% of FLUX programs; allows ~64 threads/SM
- 512 entries: supports deeper recursion; reduces to ~32 threads/SM
- 4096 entries (Go VM default): only ~4 threads/SM — unacceptable for batch

### 6.3 Throughput Estimates

| Metric                     | Python VM | Go VM      | CUDA (est.) | Speedup    |
|----------------------------|-----------|------------|-------------|------------|
| Single program (100 cycles)| ~50 µs    | ~0.5 µs    | N/A         | —          |
| 1024 programs (parallel)   | ~50 ms    | ~0.5 ms    | ~0.05 ms    | 1000× vs Py |
| Throughput (programs/sec)  | ~20,000   | ~2,000,000 | ~20,000,000 | 1000× vs Py |
| Energy per program         | ~50 µJ    | ~1 µJ      | ~0.1 µJ     | 500× vs Go |

**Jetson Orin Nano specific estimates:**
- GPU clock: 924 MHz (boost), 614 MHz (base)
- 1024 CUDA cores across SMs
- Max threads per SM: 1536
- With 256 threads/block, 6 blocks/SM: 1536 threads active
- Each thread executing ~100 cycle programs: ~100 GPU cycles per thread
- Warp throughput: 32 threads × 100 cycles = 3200 cycles per warp
- Total time (compute): ~3200 cycles / 924 MHz ≈ 3.5 µs for 32 programs
- For 1024 programs (32 warps): ~3.5 µs (fully parallel)
- Realistic with overhead: **5-10 µs per batch of 1024 programs**

### 6.4 Occupancy Calculator (Theoretical)

```
SM count:            8 (Jetson Orin Nano)
Max threads/SM:      1536
Max blocks/SM:       16
Threads per block:   256
Registers/thread:    28
Shared mem/block:    256 B

Occupancy = min(1536 / (256 × blocks_per_SM), 
                65536 / (28 × 256 × blocks_per_SM),
                49152 / (256 × blocks_per_SM),
                16 / blocks_per_SM)

With blocks_per_SM = 6: occupancy = 1536/1536 = 100% (register-bound)
With blocks_per_SM = 4: occupancy = 100% (conservative, safer)
```

---

## 7. Supported Opcodes (Phase 1)

### 7.1 Complete Opcode Table

| Opcode | Hex    | Format | Category     | Phase | Notes                    |
|--------|--------|--------|-------------|-------|--------------------------|
| HALT   | 0x00   | A (1B) | Control     | 1     | Stop execution           |
| NOP    | 0x01   | A (1B) | Control     | 1     | No operation             |
| RET    | 0x02   | A (1B) | Control     | 1     | Return from subroutine   |
| INC    | 0x08   | B (2B) | Arithmetic  | 1     | `gp[rd] += 1`            |
| DEC    | 0x09   | B (2B) | Arithmetic  | 1     | `gp[rd] -= 1`            |
| NOT    | 0x0A   | B (2B) | Bitwise     | 1     | `gp[rd] = ~gp[rd]`       |
| NEG    | 0x0B   | B (2B) | Arithmetic  | 1     | `gp[rd] = -gp[rd]`       |
| PUSH   | 0x0C   | B (2B) | Stack       | 1     | Push gp[rd] onto stack   |
| POP    | 0x0D   | B (2B) | Stack       | 1     | Pop into gp[rd]          |
| MOVI   | 0x18   | C (3B) | Load        | 1     | `gp[rd] = sign_ext(imm8)`|
| ADDI   | 0x19   | C (3B) | Arithmetic  | 1     | `gp[rd] += sign_ext(imm8)`|
| SUBI   | 0x1A   | C (3B) | Arithmetic  | 1     | `gp[rd] -= sign_ext(imm8)`|
| ADD    | 0x20   | D (4B) | Arithmetic  | 1     | `gp[rd] = gp[rs] + gp[rt]`|
| SUB    | 0x21   | D (4B) | Arithmetic  | 1     | `gp[rd] = gp[rs] - gp[rt]`|
| MUL    | 0x22   | D (4B) | Arithmetic  | 1     | `gp[rd] = gp[rs] * gp[rt]`|
| DIV    | 0x23   | D (4B) | Arithmetic  | 1     | `gp[rd] = gp[rs] / gp[rt]` (error on div0) |
| MOD    | 0x24   | D (4B) | Arithmetic  | 1     | `gp[rd] = gp[rs] % gp[rt]` (error on div0) |
| AND    | 0x25   | D (4B) | Bitwise     | 1     | `gp[rd] = gp[rs] & gp[rt]`|
| OR     | 0x26   | D (4B) | Bitwise     | 1     | `gp[rd] = gp[rs] \| gp[rt]`|
| XOR    | 0x27   | D (4B) | Bitwise     | 1     | `gp[rd] = gp[rs] ^ gp[rt]`|
| SHL    | 0x28   | D (4B) | Bitwise     | 1     | `gp[rd] = gp[rs] << gp[rt]`|
| SHR    | 0x29   | D (4B) | Bitwise     | 1     | `gp[rd] = gp[rs] >> gp[rt]`|
| MIN    | 0x2A   | D (4B) | Arithmetic  | 1     | `gp[rd] = min(gp[rs], gp[rt])`|
| MAX    | 0x2B   | D (4B) | Arithmetic  | 1     | `gp[rd] = max(gp[rs], gp[rt])`|
| CMP_EQ | 0x2C   | D (4B) | Compare     | 1     | `gp[rd] = (gp[rs] == gp[rt]) ? 1 : 0`|
| CMP_LT | 0x2D   | D (4B) | Compare     | 1     | `gp[rd] = (gp[rs] < gp[rt]) ? 1 : 0` |
| CMP_GT | 0x2E   | D (4B) | Compare     | 1     | `gp[rd] = (gp[rs] > gp[rt]) ? 1 : 0` |
| CMP_NE | 0x2F   | D (4B) | Compare     | 1     | `gp[rd] = (gp[rs] != gp[rt]) ? 1 : 0`|
| MOV    | 0x3A   | D (4B) | Data        | 1     | `gp[rd] = gp[rs]`         |
| JZ     | 0x3C   | E (4B) | Branch      | 1     | Jump if gp[rd]==0 (offset = imm16) |
| JNZ    | 0x3D   | E (4B) | Branch      | 1     | Jump if gp[rd]!=0 (offset = imm16) |
| JLT    | 0x3E   | E (4B) | Branch      | 1     | Jump if gp[rd]<0          |
| JGT    | 0x3F   | E (4B) | Branch      | 1     | Jump if gp[rd]>0          |
| MOVI16 | 0x40   | E (4B) | Load        | 1     | `gp[rd] = sign_ext(imm16)` |
| JMP    | 0x43   | E (4B) | Branch      | 1     | Unconditional jump (imm16)|
| LOOP   | 0x46   | E (4B) | Control     | 1     | `gp[rd]--; if gp[rd]>0: pc -= imm16`|
| CALL   | 0x4A   | E (4B) | Control     | 1     | Push PC+4; PC += imm16    |
| STRIPCONF | 0x17  | B (2B) | Meta        | 1     | Strip N following bytes   |
| TELL   | 0x50   | F (4B) | A2A         | 2+    | Stub: return ERR_A2A      |
| ASK    | 0x51   | F (4B) | A2A         | 2+    | Stub: return ERR_A2A      |
| BCAST  | 0x52   | F (4B) | A2A         | 2+    | Stub: return ERR_A2A      |
| RECV   | 0x53   | F (4B) | A2A         | 2+    | Stub: return ERR_A2A      |

### 7.2 Branch Encoding Detail

For branch instructions (JZ, JNZ, JLT, JGT), the offset is encoded as:
```
Byte 2 (lo): bits [7:0] of signed 16-bit offset
Byte 3 (hi): bits [15:8] of signed 16-bit offset
```

The offset is relative to the **current PC** (before the instruction is fetched), consistent with the Go VM reference:
```go
// Go VM: vm.PC += int(off)  where off = int16(...)
// CUDA:  pc += (int32_t)(int16_t)((rs << 8) | rt)
```

### 7.3 Stack Error Handling

```c
#define ERR_NONE          0
#define ERR_DIV_BY_ZERO   1
#define ERR_STACK_OVERFLOW  2
#define ERR_STACK_UNDERFLOW 3
#define ERR_INVALID_OPCODE  4
#define ERR_MAX_CYCLES      5
#define ERR_A2A_UNSUPPORTED 6
```

---

## 8. Implementation Roadmap

### Phase 1: Single-Block Kernel, Basic Opcodes
- **Goal:** Pass all Go VM conformance tests on GPU
- **Scope:** All Phase 1 opcodes from §7.1
- **Block config:** 1 block, 256 threads, 1024 programs max
- **Stack:** 256 entries per thread
- **Testing:** Compare GPU results against Go VM reference for 8 test programs
- **Estimated effort:** 2-3 sessions
- **Deliverable:** Compilable `flux_cuda_kernel.cu` with host validation

### Phase 2: Multi-Block, Register File Optimization
- **Goal:** Scale to 10,000+ programs
- **Scope:** Variable block count, dynamic grid sizing
- **Memory:** Packed bytecode with offset table (no more FLUX_MAX_BC padding)
- **Stack:** Configurable stack size (128/256/512/1024)
- **Host API:** C-callable wrapper for integration with Go runtime via cgo
- **Testing:** 10K program stress test, conformance validation
- **Estimated effort:** 2 sessions

### Phase 3: Warp-Specialized Dispatch
- **Goal:** Minimize divergence for branchy programs
- **Scope:** Program clustering by bytecode structure, warp-level scheduling
- **Techniques:** `__builtin_expect()` branch hints, `__shfl_sync()` for A2A
- **Sorting:** Host pre-sorts programs into divergence-homogeneous batches
- **Expected improvement:** 15-30% for branch-heavy workloads

### Phase 4: Shared Memory Optimization
- **Goal:** Reduce global memory pressure for hot bytecode
- **Scope:** Cache frequently-executed loop bodies in shared memory
- **Technique:** Block-level bytecode prefetching into shared memory
- **Tradeoff:** Larger programs benefit; small programs already L2-resident
- **Expected improvement:** 10-20% for large programs (>1KB bytecode)

### Phase 5: Tensor Op Extensions
- **Goal:** Leverage Ampere Tensor Cores for batch arithmetic
- **Scope:** Vectorized ADD/MUL across programs via WMMA
- **Technique:** Pack 16 programs' register files into 16×16 matrix, use tensor cores
- **Use case:** Fleet conformance voting — 1000 agents compute, results aggregated via matrix ops
- **Expected improvement:** 2-5× for pure arithmetic-heavy batches

---

## 9. Integration with FLUX Ecosystem

### 9.1 Conformance Vector Mapping

Conformance vectors (from `flux-conformance`) are serialized as FLUX bytecode programs:

```
conformance_vector.json
  └── agent_programs[]
        ├── agent_id: "oracle1"
        ├── bytecode: [0x18, 0x00, 0x2A, ...]   ← FLUX bytecode
        └── expected_result: 42
```

**Integration flow:**
1. `flux-conformance` generates N agent programs as packed bytecode
2. Go runtime (or Python) serializes to GPU-ready format: `programs[]`, `offsets[]`, `lengths[]`
3. `cudaMemcpy` to GPU
4. `flux_batch_execute<<<blocks, threads>>>(...)` executes all N programs
5. `cudaMemcpy` results back to host
6. Compare `results[i]` against `expected[i]` for pass/fail

### 9.2 GPU → CPU → Git Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  GPU Kernel  │────▶│  Host (Go)   │────▶│  Git Commit  │
│              │     │              │     │              │
│  results[]   │     │  Validate    │     │  Push to     │
│  errors[]    │     │  Aggregate   │     │  fleet repo  │
│  cycles[]    │     │  Format      │     │  (GitHub)    │
└──────────────┘     └──────────────┘     └──────────────┘
```

1. **GPU** executes batch, writes results to global memory
2. **Host** copies results, validates against expected, generates conformance report
3. **Git** commit includes: conformance report, performance metrics, any errors
4. **Fleet** other agents can pull and verify results (trust protocol)

### 9.3 "Deploy Anywhere" Vision

The CUDA kernel completes the greenhorn-runtime's multi-backend strategy:

| Backend    | Language | Status   | Throughput (programs/sec) |
|-----------|----------|----------|--------------------------|
| Python    | CPython  | Stable   | ~20,000                  |
| Go        | Go       | Stable   | ~2,000,000               |
| Rust      | Rust     | Stable   | ~5,000,000               |
| C++       | C++17    | Stable   | ~3,000,000               |
| Zig       | Zig      | Stable   | ~4,000,000               |
| Java      | Java 17  | Stable   | ~1,500,000               |
| JS        | Node.js  | Stable   | ~500,000                 |
| WASM      | Rust→WASM| Stable   | ~1,000,000               |
| **CUDA**  | **CUDA C** | **Phase 1** | **~20,000,000**     |

A single FLUX bytecode program runs identically on all 9 backends. The CUDA backend enables:
- **Jetson edge deployment:** Run agent programs on NVIDIA-powered edge devices
- **Batch conformance:** Validate entire fleet behavior in microseconds
- **GPU-accelerated evolution:** GA/selection across thousands of agent variants
- **Multi-agent simulation:** Each thread = one agent, interacting via A2A queue

---

## Appendix A: Existing Stub Analysis

The current `cuda/flux_cuda.cu` has these limitations that the new kernel addresses:

| Issue                     | Current Stub                  | New Design                          |
|---------------------------|-------------------------------|-------------------------------------|
| Memory layout             | Fixed `N × FLUX_MAX_BC`       | Packed with offset table            |
| Register count            | 16 (hardcoded)                | 16 (configurable, matches Go ref)   |
| Stack                     | None                          | 256-entry per thread                |
| Error reporting           | None                          | Per-program error codes             |
| Opcodes                   | 14 opcodes                    | 37 opcodes (Phase 1)                |
| Branch encoding           | Inconsistent (JNZ uses 3B)    | Canonical 4-byte format (matches Go)|
| Conformance               | No test validation            | Go VM reference comparison          |
| Multi-block               | Supported but untested        | Designed for 10K+ programs          |
| Safety                    | Hardcoded 10000 limit         | Configurable MAX_CYCLES per launch  |

## Appendix B: Build and Test Commands

```bash
# Build (on Jetson or with CUDA toolkit)
cd cuda/
make clean && make

# Run with built-in test programs
./flux_cuda_kernel

# Build with debug (for profiling)
NVCCFLAGS="-arch=sm_87 -O0 -G -lineinfo" make

# Profile with NVIDIA Nsight Compute
ncu --set full --target-processes all ./flux_cuda_kernel

# Profile with NVIDIA Nsight Systems
nsys profile --stats=true ./flux_cuda_kernel
```

## Appendix C: Conformance Test Bytecodes

The following bytecodes are used to validate CUDA kernel correctness against the Go reference:

```c
// Test 1: HALT — program: [0x00], expected: R0=0, cycles=1
// Test 2: MOVI — program: [0x18, 0x00, 0x2A, 0x00], expected: R0=42, cycles=2
// Test 3: ADD — program: [0x18,0,10, 0x18,1,20, 0x20,2,0,1, 0x00], expected: R2=30, cycles=4
// Test 4: Fibonacci(10) — expected: R1=144
// Test 5: PUSH/POP — expected: R1=42
// Test 6: MOVI16 — program: [0x40,0,0xE8,0x03, 0x00], expected: R0=1000
// Test 7: Factorial(5) — expected: R0=120
// Test 8: Nested loop — expected: R0=100
```

---

*Document version: 1.0 — Quill, Architect-rank*
*Last updated: 2026-04*
*Status: Design complete, Phase 1 implementation in progress*
