/**
 * ============================================================================
 * FLUX CUDA Batch Execution Kernel — Implementation
 * ============================================================================
 *
 * Parallel FLUX bytecode VM for NVIDIA GPU. Each CUDA thread executes
 * one independent FLUX program. This file contains both device kernels
 * and the host-side executor implementation.
 *
 * Design:    See cuda/DESIGN.md for full architecture document
 * Reference: pkg/flux/vm.go (canonical Go implementation)
 *
 * Build:     nvcc -arch=sm_87 -O2 -c batch_kernel.cu -o batch_kernel.o
 *            gcc   -c batch_kernel.cpp -o batch_kernel_cpp.o  (host fallback)
 *            g++   batch_kernel.o batch_kernel_cpp.o -lcudart -o libfluxbatch.so
 *
 * Author:    T-005 (greenhorn-runtime)
 * ============================================================================
 */

#include "batch_executor.cuh"

#ifdef FLUX_CUDA_AVAILABLE
#include <cuda_runtime.h>

// ============================================================================
// Section 1: FLUX Opcode Definitions (Canonical ISA)
// ============================================================================

// --- Format A: 1-byte instructions (0x00-0x07) ---
#define OP_HALT   0x00
#define OP_NOP    0x01
#define OP_RET    0x02

// --- Format B: 2-byte instructions (0x08-0x17) ---
#define OP_INC       0x08
#define OP_DEC       0x09
#define OP_NOT       0x0A
#define OP_NEG       0x0B
#define OP_PUSH      0x0C
#define OP_POP       0x0D
#define OP_STRIPCONF 0x17

// --- Format C: 3-byte instructions (0x18-0x1F) ---
#define OP_MOVI  0x18
#define OP_ADDI  0x19
#define OP_SUBI  0x1A

// --- Format D: 4-byte instructions — 3-register (0x20-0x3F) ---
#define OP_ADD    0x20
#define OP_SUB    0x21
#define OP_MUL    0x22
#define OP_DIV    0x23
#define OP_MOD    0x24
#define OP_AND    0x25
#define OP_OR     0x26
#define OP_XOR    0x27
#define OP_SHL    0x28
#define OP_SHR    0x29
#define OP_MIN    0x2A
#define OP_MAX    0x2B
#define OP_CMP_EQ 0x2C
#define OP_CMP_LT 0x2D
#define OP_CMP_GT 0x2E
#define OP_CMP_NE 0x2F
#define OP_MOV    0x3A
#define OP_JZ     0x3C
#define OP_JNZ    0x3D
#define OP_JLT    0x3E
#define OP_JGT    0x3F

// --- Format E: 4-byte instructions — register + immediate16 (0x40-0x4F) ---
#define OP_MOVI16 0x40
#define OP_JMP    0x43
#define OP_LOOP   0x46
#define OP_CALL   0x4A

// --- Format F: A2A (0x50+) ---
#define OP_TELL   0x50
#define OP_ASK    0x51
#define OP_BCAST  0x52
#define OP_RECV   0x53

// ============================================================================
// Section 2: Constant Memory — Opcode Format Table
// ============================================================================

/**
 * Pre-computed instruction byte-lengths for each opcode.
 * Loaded into __constant__ memory (64KB, broadcast to warp in ~4 cycles).
 * Matches Go vm.formatSize() exactly.
 */
__constant__ uint8_t opcode_format[256] = {
    // 0x00-0x07: Format A (1 byte)
    1, 1, 1, 1, 1, 1, 1, 1,
    // 0x08-0x0F: Format B (2 bytes)
    2, 2, 2, 2, 2, 2, 2, 2,
    // 0x10-0x17: Format B' (2 bytes)
    2, 2, 2, 2, 2, 2, 2, 2,
    // 0x18-0x1F: Format C (3 bytes)
    3, 3, 3, 3, 3, 3, 3, 3,
    // 0x20-0x2F: Format D (4 bytes)
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    // 0x30-0x3F: Format D (4 bytes)
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    // 0x40-0x4F: Format E (4 bytes)
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    // 0x50-0xFF: 1 byte each (A2A stubs, unassigned)
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
};

// ============================================================================
// Section 3: Device Helper Functions
// ============================================================================

/** Decode signed 8-bit immediate. Matches Go: int32(int8(bc[pc+2])) */
__device__ __forceinline__ int32_t decode_imm8(uint8_t b) {
    return (int32_t)(int8_t)b;
}

/** Decode signed 16-bit immediate (little-endian). Matches Go: int32(int16(...)) */
__device__ __forceinline__ int32_t decode_imm16(uint8_t lo, uint8_t hi) {
    return (int32_t)(int16_t)((uint16_t)lo | ((uint16_t)hi << 8));
}

/** Validate register index is within GPU register file bounds. */
__device__ __forceinline__ bool valid_reg(uint8_t rd) {
    return rd < FLUX_NUM_REGS;
}

// ============================================================================
// Section 4: Main Execution Kernel
// ============================================================================

/**
 * flux_batch_execute_kernel — Execute N FLUX programs in parallel on GPU.
 *
 * Thread mapping: global_id = blockIdx.x * blockDim.x + threadIdx.x
 * Each thread with global_id < num_programs executes one FLUX program.
 *
 * Per-thread state (all in registers or local memory):
 *   - 16 GP registers (64 bytes, in hardware registers)
 *   - 256-entry stack (1KB, in local memory, L1/L2 cached)
 *   - PC, SP, cycles, flags (scalars, in hardware registers)
 *
 * @param programs     Packed bytecode array (all programs concatenated)
 * @param offsets      Start offset for each program in the packed array
 * @param lengths      Bytecode length for each program
 * @param results      Output: GP[0] per program
 * @param error_codes  Output: error code per program (0=success)
 * @param cycle_counts Output: total cycles consumed per program
 * @param num_programs Total programs in this batch
 */
__global__ void flux_batch_execute_kernel(
    const uint8_t*  __restrict__ programs,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ lengths,
    int32_t*        __restrict__ results,
    int32_t*        __restrict__ error_codes,
    int32_t*        __restrict__ cycle_counts,
    int num_programs
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Out-of-bounds threads exit (no divergence — at end of last warp)
    if (global_id >= num_programs) return;

    // --- Per-thread VM state ---
    int32_t gp[FLUX_NUM_REGS];
    int32_t stack[FLUX_STACK_SIZE];
    int32_t sp       = FLUX_STACK_SIZE;
    int32_t pc       = 0;
    int32_t cycles   = 0;
    int     halted   = 0;
    int     error    = FLUX_ERR_NONE;
    int     branched = 0;

    // Zero-init registers
    #pragma unroll
    for (int i = 0; i < FLUX_NUM_REGS; i++) {
        gp[i] = 0;
    }

    // Load program metadata (coalesced global memory read)
    uint32_t base_offset = offsets[global_id];
    uint32_t prog_len    = lengths[global_id];
    const uint8_t* bc    = programs + base_offset;

    // ================================================================
    // MAIN DISPATCH LOOP
    // ================================================================
    while (!halted && pc >= 0 && (uint32_t)pc < prog_len && cycles < FLUX_MAX_CYCLES) {
        uint8_t opcode = bc[pc];
        uint8_t fmt    = opcode_format[opcode];

        // Boundary check
        if ((uint32_t)(pc + fmt) > prog_len) {
            error = FLUX_ERR_PC_OOB;
            break;
        }

        // Fetch operands
        uint8_t rd = 0, rs = 0, rt = 0;
        if (fmt >= 2) rd = bc[pc + 1];
        if (fmt >= 3) rs = bc[pc + 2];
        if (fmt >= 4) rt = bc[pc + 3];

        cycles++;
        branched = 0;

        // --- Dispatch via switch (NVCC generates jump table) ---
        switch (opcode) {

            // FORMAT A: 1-byte (no operands)
            case OP_HALT:
                halted = 1;
                break;
            case OP_NOP:
                break;
            case OP_RET:
                if (__builtin_expect(sp >= FLUX_STACK_SIZE, 0)) {
                    error = FLUX_ERR_STACK_UNDERFLOW;
                    break;
                }
                pc = stack[sp];
                sp++;
                branched = 1;
                break;

            // FORMAT B: 2-byte [opcode, rd]
            case OP_INC:
                if (__builtin_expect(valid_reg(rd), 1)) gp[rd] += 1;
                else error = FLUX_ERR_BAD_REGISTER;
                break;
            case OP_DEC:
                if (__builtin_expect(valid_reg(rd), 1)) gp[rd] -= 1;
                else error = FLUX_ERR_BAD_REGISTER;
                break;
            case OP_NOT:
                if (__builtin_expect(valid_reg(rd), 1)) gp[rd] = ~gp[rd];
                else error = FLUX_ERR_BAD_REGISTER;
                break;
            case OP_NEG:
                if (__builtin_expect(valid_reg(rd), 1)) gp[rd] = -gp[rd];
                else error = FLUX_ERR_BAD_REGISTER;
                break;
            case OP_PUSH:
                if (__builtin_expect(valid_reg(rd), 1)) {
                    if (__builtin_expect(sp > 0, 1)) {
                        sp--;
                        stack[sp] = gp[rd];
                    } else {
                        error = FLUX_ERR_STACK_OVERFLOW;
                    }
                } else {
                    error = FLUX_ERR_BAD_REGISTER;
                }
                break;
            case OP_POP:
                if (__builtin_expect(valid_reg(rd), 1)) {
                    if (__builtin_expect(sp < FLUX_STACK_SIZE, 1)) {
                        gp[rd] = stack[sp];
                        sp++;
                    } else {
                        error = FLUX_ERR_STACK_UNDERFLOW;
                    }
                } else {
                    error = FLUX_ERR_BAD_REGISTER;
                }
                break;
            case OP_STRIPCONF:
                break;

            // FORMAT C: 3-byte [opcode, rd, imm8]
            case OP_MOVI:
                if (__builtin_expect(valid_reg(rd), 1)) gp[rd] = decode_imm8(rs);
                else error = FLUX_ERR_BAD_REGISTER;
                break;
            case OP_ADDI:
                if (__builtin_expect(valid_reg(rd), 1)) gp[rd] += decode_imm8(rs);
                else error = FLUX_ERR_BAD_REGISTER;
                break;
            case OP_SUBI:
                if (__builtin_expect(valid_reg(rd), 1)) gp[rd] -= decode_imm8(rs);
                else error = FLUX_ERR_BAD_REGISTER;
                break;

            // FORMAT D: 4-byte [opcode, rd, rs, rt] — Arithmetic
            case OP_ADD:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1))
                    gp[rd] = gp[rs] + gp[rt];
                else error = FLUX_ERR_BAD_REGISTER;
                break;
            case OP_SUB:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1))
                    gp[rd] = gp[rs] - gp[rt];
                else error = FLUX_ERR_BAD_REGISTER;
                break;
            case OP_MUL:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1))
                    gp[rd] = gp[rs] * gp[rt];
                else error = FLUX_ERR_BAD_REGISTER;
                break;
            case OP_DIV:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1)) {
                    if (__builtin_expect(gp[rt] != 0, 1))
                        gp[rd] = gp[rs] / gp[rt];
                    else
                        error = FLUX_ERR_DIV_BY_ZERO;
                } else {
                    error = FLUX_ERR_BAD_REGISTER;
                }
                break;
            case OP_MOD:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1)) {
                    if (__builtin_expect(gp[rt] != 0, 1))
                        gp[rd] = gp[rs] % gp[rt];
                    else
                        error = FLUX_ERR_DIV_BY_ZERO;
                } else {
                    error = FLUX_ERR_BAD_REGISTER;
                }
                break;

            // FORMAT D: Bitwise
            case OP_AND:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1))
                    gp[rd] = gp[rs] & gp[rt];
                else error = FLUX_ERR_BAD_REGISTER;
                break;
            case OP_OR:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1))
                    gp[rd] = gp[rs] | gp[rt];
                else error = FLUX_ERR_BAD_REGISTER;
                break;
            case OP_XOR:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1))
                    gp[rd] = gp[rs] ^ gp[rt];
                else error = FLUX_ERR_BAD_REGISTER;
                break;
            case OP_SHL:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1))
                    gp[rd] = gp[rs] << (gp[rt] & 31);
                else error = FLUX_ERR_BAD_REGISTER;
                break;
            case OP_SHR:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1))
                    gp[rd] = gp[rs] >> (gp[rt] & 31);
                else error = FLUX_ERR_BAD_REGISTER;
                break;

            // FORMAT D: Min/Max
            case OP_MIN:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1))
                    gp[rd] = (gp[rs] < gp[rt]) ? gp[rs] : gp[rt];
                else error = FLUX_ERR_BAD_REGISTER;
                break;
            case OP_MAX:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1))
                    gp[rd] = (gp[rs] > gp[rt]) ? gp[rs] : gp[rt];
                else error = FLUX_ERR_BAD_REGISTER;
                break;

            // FORMAT D: Comparison
            case OP_CMP_EQ:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1))
                    gp[rd] = (gp[rs] == gp[rt]) ? 1 : 0;
                else error = FLUX_ERR_BAD_REGISTER;
                break;
            case OP_CMP_LT:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1))
                    gp[rd] = (gp[rs] < gp[rt]) ? 1 : 0;
                else error = FLUX_ERR_BAD_REGISTER;
                break;
            case OP_CMP_GT:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1))
                    gp[rd] = (gp[rs] > gp[rt]) ? 1 : 0;
                else error = FLUX_ERR_BAD_REGISTER;
                break;
            case OP_CMP_NE:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1))
                    gp[rd] = (gp[rs] != gp[rt]) ? 1 : 0;
                else error = FLUX_ERR_BAD_REGISTER;
                break;

            // FORMAT D: Data movement
            case OP_MOV:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs), 1))
                    gp[rd] = gp[rs];
                else error = FLUX_ERR_BAD_REGISTER;
                break;

            // FORMAT D/E: Conditional branches (causes warp divergence)
            // NOTE: JZ/JNZ/JLT/JGT use int8 offset (byte rs) to match the
            // Go VM reference (pkg/flux/vm.go). Byte rt is padding.
            // JMP/MOVI16/LOOP/CALL use int16 offsets (rs|rt<<8).
            case OP_JZ:
                if (__builtin_expect(valid_reg(rd), 1)) {
                    int32_t offset = decode_imm8(rs);
                    if (__builtin_expect(gp[rd] == 0, 0)) {
                        pc += offset;
                        branched = 1;
                    }
                } else {
                    error = FLUX_ERR_BAD_REGISTER;
                }
                break;
            case OP_JNZ:
                if (__builtin_expect(valid_reg(rd), 1)) {
                    int32_t offset = decode_imm8(rs);
                    if (__builtin_expect(gp[rd] != 0, 1)) {
                        pc += offset;
                        branched = 1;
                    }
                } else {
                    error = FLUX_ERR_BAD_REGISTER;
                }
                break;
            case OP_JLT:
                if (__builtin_expect(valid_reg(rd), 1)) {
                    int32_t offset = decode_imm8(rs);
                    if (gp[rd] < 0) { pc += offset; branched = 1; }
                } else {
                    error = FLUX_ERR_BAD_REGISTER;
                }
                break;
            case OP_JGT:
                if (__builtin_expect(valid_reg(rd), 1)) {
                    int32_t offset = decode_imm8(rs);
                    if (gp[rd] > 0) { pc += offset; branched = 1; }
                } else {
                    error = FLUX_ERR_BAD_REGISTER;
                }
                break;

            // FORMAT E: Register + immediate16
            case OP_MOVI16:
                if (__builtin_expect(valid_reg(rd), 1))
                    gp[rd] = decode_imm16(rs, rt);
                else error = FLUX_ERR_BAD_REGISTER;
                break;
            case OP_JMP:
                {
                    int32_t offset = decode_imm16(rs, rt);
                    pc += offset;
                    branched = 1;
                }
                break;
            case OP_LOOP:
                if (__builtin_expect(valid_reg(rd), 1)) {
                    uint16_t back_off = (uint16_t)rs | ((uint16_t)rt << 8);
                    gp[rd] -= 1;
                    if (__builtin_expect(gp[rd] > 0, 1)) {
                        pc -= (int32_t)back_off;
                        branched = 1;
                    }
                } else {
                    error = FLUX_ERR_BAD_REGISTER;
                }
                break;
            case OP_CALL:
                if (__builtin_expect(valid_reg(rd), 1)) {
                    if (__builtin_expect(sp > 0, 1)) {
                        sp--;
                        stack[sp] = pc + 4;
                        int32_t offset = decode_imm16(rs, rt);
                        pc += offset;
                        branched = 1;
                    } else {
                        error = FLUX_ERR_STACK_OVERFLOW;
                    }
                } else {
                    error = FLUX_ERR_BAD_REGISTER;
                }
                break;

            // FORMAT F: A2A (stubbed — Phase 2+)
            case OP_TELL:
            case OP_ASK:
            case OP_BCAST:
            case OP_RECV:
                error = FLUX_ERR_A2A_UNSUPPORTED;
                halted = 1;
                break;

            default:
                break;
        }

        // Advance PC unless branch set it
        if (!halted && !branched) {
            pc += (int32_t)fmt;
        }

        // Halt on error (except A2A which already set halted)
        if (error != FLUX_ERR_NONE && error != FLUX_ERR_A2A_UNSUPPORTED) {
            halted = 1;
        }
    }

    // Max cycles check
    if (!halted && cycles >= FLUX_MAX_CYCLES) {
        error = FLUX_ERR_MAX_CYCLES;
    }

    // Write results (coalesced global memory writes)
    results[global_id]      = gp[0];
    error_codes[global_id]  = error;
    cycle_counts[global_id] = cycles;
}

#endif // FLUX_CUDA_AVAILABLE

// ============================================================================
// Section 5: Host-Side CPU Reference Implementation (Fallback)
// ============================================================================
//
// When CUDA is not available, batch_kernel.cu provides a pure-C
// reference implementation that produces identical results. This allows:
//   - Testing the Go bindings without a GPU
//   - CI environments without CUDA toolkits
//   - Conformance validation of the CPU path against Go VM
// ============================================================================

#ifndef FLUX_CUDA_AVAILABLE

/**
 * CPU fallback: Execute a single FLUX program step.
 * Matches the Go VM (pkg/flux/vm.go) instruction semantics exactly.
 *
 * @param bc       Bytecode array
 * @param len      Bytecode length
 * @param pc       [in/out] Program counter
 * @param gp       Register file (16 entries)
 * @param stack    Stack array
 * @param sp       [in/out] Stack pointer
 * @param halted   [in/out] Halt flag
 * @return Error code (0 = continue, negative = stop)
 */
static int cpu_flux_step(
    const uint8_t* bc, int len, int* pc,
    int32_t* gp, int32_t* stack, int* sp, int* halted
) {
    if (*pc >= len || *halted) return -1;

    uint8_t op = bc[*pc];
    int size;

    // Format size lookup (matches Go formatSize)
    if (op <= 0x07) size = 1;
    else if (op <= 0x17) size = 2;
    else if (op <= 0x1F) size = 3;
    else if (op <= 0x4F) size = 4;
    else size = 1;

    if (*pc + size > len) { *halted = 1; return -1; }

    switch (op) {
        case OP_HALT: *halted = 1; return 1;
        case OP_NOP: return 1;
        case OP_RET:
            if (*sp >= FLUX_STACK_SIZE) return -1;
            *pc = stack[*sp]; (*sp)++;
            return 0; // PC already set
        case OP_INC: gp[bc[*pc+1]]++; return 2;
        case OP_DEC: gp[bc[*pc+1]]--; return 2;
        case OP_NOT: gp[bc[*pc+1]] = ~gp[bc[*pc+1]]; return 2;
        case OP_NEG: gp[bc[*pc+1]] = -gp[bc[*pc+1]]; return 2;
        case OP_PUSH:
            (*sp)--;
            stack[*sp] = gp[bc[*pc+1]];
            return 2;
        case OP_POP:
            gp[bc[*pc+1]] = stack[*sp];
            (*sp)++;
            return 2;
        case OP_STRIPCONF: return 2;
        case OP_MOVI: gp[bc[*pc+1]] = (int32_t)(int8_t)bc[*pc+2]; return 3;
        case OP_ADDI: gp[bc[*pc+1]] += (int32_t)(int8_t)bc[*pc+2]; return 3;
        case OP_SUBI: gp[bc[*pc+1]] -= (int32_t)(int8_t)bc[*pc+2]; return 3;
        case OP_ADD: gp[bc[*pc+1]] = gp[bc[*pc+2]] + gp[bc[*pc+3]]; return 4;
        case OP_SUB: gp[bc[*pc+1]] = gp[bc[*pc+2]] - gp[bc[*pc+3]]; return 4;
        case OP_MUL: gp[bc[*pc+1]] = gp[bc[*pc+2]] * gp[bc[*pc+3]]; return 4;
        case OP_DIV:
            if (gp[bc[*pc+3]] == 0) { *halted = 1; return 4; }
            gp[bc[*pc+1]] = gp[bc[*pc+2]] / gp[bc[*pc+3]];
            return 4;
        case OP_MOD:
            if (gp[bc[*pc+3]] == 0) { *halted = 1; return 4; }
            gp[bc[*pc+1]] = gp[bc[*pc+2]] % gp[bc[*pc+3]];
            return 4;
        case OP_AND: gp[bc[*pc+1]] = gp[bc[*pc+2]] & gp[bc[*pc+3]]; return 4;
        case OP_OR:  gp[bc[*pc+1]] = gp[bc[*pc+2]] | gp[bc[*pc+3]]; return 4;
        case OP_XOR: gp[bc[*pc+1]] = gp[bc[*pc+2]] ^ gp[bc[*pc+3]]; return 4;
        case OP_SHL: gp[bc[*pc+1]] = gp[bc[*pc+2]] << (gp[bc[*pc+3]] & 31); return 4;
        case OP_SHR: gp[bc[*pc+1]] = gp[bc[*pc+2]] >> (gp[bc[*pc+3]] & 31); return 4;
        case OP_MIN: gp[bc[*pc+1]] = (gp[bc[*pc+2]] < gp[bc[*pc+3]]) ? gp[bc[*pc+2]] : gp[bc[*pc+3]]; return 4;
        case OP_MAX: gp[bc[*pc+1]] = (gp[bc[*pc+2]] > gp[bc[*pc+3]]) ? gp[bc[*pc+2]] : gp[bc[*pc+3]]; return 4;
        case OP_CMP_EQ: gp[bc[*pc+1]] = (gp[bc[*pc+2]] == gp[bc[*pc+3]]) ? 1 : 0; return 4;
        case OP_CMP_LT: gp[bc[*pc+1]] = (gp[bc[*pc+2]] < gp[bc[*pc+3]]) ? 1 : 0; return 4;
        case OP_CMP_GT: gp[bc[*pc+1]] = (gp[bc[*pc+2]] > gp[bc[*pc+3]]) ? 1 : 0; return 4;
        case OP_CMP_NE: gp[bc[*pc+1]] = (gp[bc[*pc+2]] != gp[bc[*pc+3]]) ? 1 : 0; return 4;
        case OP_MOV: gp[bc[*pc+1]] = gp[bc[*pc+2]]; return 4;
        case OP_JNZ: {
            uint8_t rd = bc[*pc+1];
            int32_t off = (int32_t)(int8_t)bc[*pc+2];
            if (gp[rd] != 0) { *pc += off; return 0; }
            return 4;
        }
        case OP_JZ: {
            uint8_t rd = bc[*pc+1];
            int32_t off = (int32_t)(int8_t)bc[*pc+2];
            if (gp[rd] == 0) { *pc += off; return 0; }
            return 4;
        }
        case OP_MOVI16: {
            int16_t imm = (int16_t)((uint16_t)bc[*pc+2] | ((uint16_t)bc[*pc+3] << 8));
            gp[bc[*pc+1]] = (int32_t)imm;
            return 4;
        }
        case OP_JMP: {
            int16_t off = (int16_t)((uint16_t)bc[*pc+2] | ((uint16_t)bc[*pc+3] << 8));
            *pc += off;
            return 0;
        }
        case OP_LOOP: {
            uint8_t rd = bc[*pc+1];
            int off = (int)((uint16_t)bc[*pc+2] | ((uint16_t)bc[*pc+3] << 8));
            gp[rd]--;
            if (gp[rd] > 0) { *pc -= off; return 0; }
            return 4;
        }
        default: return size;
    }
}

#endif // !FLUX_CUDA_AVAILABLE

// ============================================================================
// Section 6: Host-Side Executor Implementation
// ============================================================================

#include <stdlib.h>
#include <string.h>

// Opaque handle definition
struct FluxBatchHandle {
    // CUDA-specific (only used when FLUX_CUDA_AVAILABLE)
#ifdef FLUX_CUDA_AVAILABLE
    cudaStream_t   stream;
    uint8_t*       d_programs;
    uint32_t*      d_offsets;
    uint32_t*      d_lengths;
    int32_t*       d_results;
    int32_t*       d_errors;
    int32_t*       d_cycles;
    size_t         programs_capacity;
    int            programs_capacity_count;
    cudaEvent_t    start_event;
    cudaEvent_t    stop_event;
#endif
    int            block_size;
    int            max_cycles;
    int            device_id;
    char           error_msg[256];
};

// ============================================================================
// Lifecycle
// ============================================================================

FluxBatchHandle* flux_batch_init(void) {
    FluxBatchConfig config;
    config.block_size = FLUX_CUDA_BLOCK_SIZE;
    config.max_cycles = FLUX_MAX_CYCLES;
    config.device_id  = 0;
    return flux_batch_init_ex(&config);
}

FluxBatchHandle* flux_batch_init_ex(const FluxBatchConfig* config) {
    FluxBatchHandle* h = (FluxBatchHandle*)calloc(1, sizeof(FluxBatchHandle));
    if (!h) return NULL;

    h->block_size = config ? config->block_size : FLUX_CUDA_BLOCK_SIZE;
    h->max_cycles = config ? config->max_cycles : FLUX_MAX_CYCLES;
    h->device_id  = config ? config->device_id  : 0;
    strncpy(h->error_msg, "no error", sizeof(h->error_msg));

#ifdef FLUX_CUDA_AVAILABLE
    // Set device
    cudaError_t err = cudaSetDevice(h->device_id);
    if (err != cudaSuccess) {
        snprintf(h->error_msg, sizeof(h->error_msg),
                 "cudaSetDevice(%d) failed: %s", h->device_id, cudaGetErrorString(err));
        free(h);
        return NULL;
    }

    // Create stream
    err = cudaStreamCreate(&h->stream);
    if (err != cudaSuccess) {
        snprintf(h->error_msg, sizeof(h->error_msg),
                 "cudaStreamCreate failed: %s", cudaGetErrorString(err));
        free(h);
        return NULL;
    }

    // Create timing events
    cudaEventCreate(&h->start_event);
    cudaEventCreate(&h->stop_event);

    // Device pointers start as NULL (allocated on first use)
    h->d_programs = NULL;
    h->d_offsets  = NULL;
    h->d_lengths  = NULL;
    h->d_results  = NULL;
    h->d_errors   = NULL;
    h->d_cycles   = NULL;
    h->programs_capacity = 0;
    h->programs_capacity_count = 0;
#endif

    return h;
}

void flux_batch_destroy(FluxBatchHandle* handle) {
    if (!handle) return;

#ifdef FLUX_CUDA_AVAILABLE
    if (handle->d_programs) cudaFree(handle->d_programs);
    if (handle->d_offsets)  cudaFree(handle->d_offsets);
    if (handle->d_lengths)  cudaFree(handle->d_lengths);
    if (handle->d_results)  cudaFree(handle->d_results);
    if (handle->d_errors)   cudaFree(handle->d_errors);
    if (handle->d_cycles)   cudaFree(handle->d_cycles);
    cudaEventDestroy(handle->start_event);
    cudaEventDestroy(handle->stop_event);
    cudaStreamDestroy(handle->stream);
#endif

    free(handle);
}

const char* flux_batch_get_error(FluxBatchHandle* handle) {
    if (!handle) return "null handle";
    return handle->error_msg;
}

// ============================================================================
// Execution
// ============================================================================

FluxBatchResult* flux_batch_run(
    FluxBatchHandle*      handle,
    const uint8_t*        programs,
    const uint32_t*       offsets,
    const uint32_t*       lengths,
    int                   num_programs,
    size_t                total_bc_size
) {
    if (!handle || !programs || !offsets || !lengths || num_programs <= 0) {
        if (handle) snprintf(handle->error_msg, sizeof(handle->error_msg), "invalid arguments");
        return NULL;
    }

    // Allocate result struct
    FluxBatchResult* r = (FluxBatchResult*)calloc(1, sizeof(FluxBatchResult));
    if (!r) {
        snprintf(handle->error_msg, sizeof(handle->error_msg), "out of memory");
        return NULL;
    }
    r->num_programs = num_programs;
    r->h_results = (int32_t*)calloc(num_programs, sizeof(int32_t));
    r->h_errors  = (int32_t*)calloc(num_programs, sizeof(int32_t));
    r->h_cycles  = (int32_t*)calloc(num_programs, sizeof(int32_t));
    if (!r->h_results || !r->h_errors || !r->h_cycles) {
        snprintf(handle->error_msg, sizeof(handle->error_msg), "out of memory for results");
        free(r->h_results); free(r->h_errors); free(r->h_cycles); free(r);
        return NULL;
    }

#ifdef FLUX_CUDA_AVAILABLE
    // --- GPU path ---
    cudaSetDevice(handle->device_id);

    // Allocate or reallocate GPU buffers if needed
    if ((size_t)num_programs > handle->programs_capacity ||
        total_bc_size > handle->programs_capacity) {

        if (handle->d_programs) cudaFree(handle->d_programs);
        if (handle->d_offsets)  cudaFree(handle->d_offsets);
        if (handle->d_lengths)  cudaFree(handle->d_lengths);
        if (handle->d_results)  cudaFree(handle->d_results);
        if (handle->d_errors)   cudaFree(handle->d_errors);
        if (handle->d_cycles)   cudaFree(handle->d_cycles);

        cudaError_t err;
        err = cudaMalloc(&handle->d_programs, total_bc_size);
        if (err != cudaSuccess) goto gpu_error;
        err = cudaMalloc(&handle->d_offsets, num_programs * sizeof(uint32_t));
        if (err != cudaSuccess) goto gpu_error;
        err = cudaMalloc(&handle->d_lengths, num_programs * sizeof(uint32_t));
        if (err != cudaSuccess) goto gpu_error;
        err = cudaMalloc(&handle->d_results, num_programs * sizeof(int32_t));
        if (err != cudaSuccess) goto gpu_error;
        err = cudaMalloc(&handle->d_errors, num_programs * sizeof(int32_t));
        if (err != cudaSuccess) goto gpu_error;
        err = cudaMalloc(&handle->d_cycles, num_programs * sizeof(int32_t));
        if (err != cudaSuccess) goto gpu_error;

        handle->programs_capacity = total_bc_size;
        handle->programs_capacity_count = num_programs;
    }

    // Copy data to GPU (async on stream)
    cudaMemcpyAsync(handle->d_programs, programs, total_bc_size,
                    cudaMemcpyHostToDevice, handle->stream);
    cudaMemcpyAsync(handle->d_offsets, offsets, num_programs * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, handle->stream);
    cudaMemcpyAsync(handle->d_lengths, lengths, num_programs * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, handle->stream);

    // Compute grid dimensions
    int threads = handle->block_size;
    int blocks  = (num_programs + threads - 1) / threads;

    // Launch kernel with timing
    cudaEventRecord(handle->start_event, handle->stream);
    flux_batch_execute_kernel<<<blocks, threads, 0, handle->stream>>>(
        handle->d_programs, handle->d_offsets, handle->d_lengths,
        handle->d_results, handle->d_errors, handle->d_cycles,
        num_programs
    );
    cudaEventRecord(handle->stop_event, handle->stream);

    // Synchronize and measure
    cudaStreamSynchronize(handle->stream);
    cudaEventElapsedTime(&r->gpu_ms, handle->start_event, handle->stop_event);

    // Copy results back
    cudaMemcpyAsync(r->h_results, handle->d_results, num_programs * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, handle->stream);
    cudaMemcpyAsync(r->h_errors, handle->d_errors, num_programs * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, handle->stream);
    cudaMemcpyAsync(r->h_cycles, handle->d_cycles, num_programs * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, handle->stream);
    cudaStreamSynchronize(handle->stream);

    r->cuda_error = 0;
    r->results = 0; // Device pointer not exposed to Go
    return r;

gpu_error:
    snprintf(handle->error_msg, sizeof(handle->error_msg),
             "CUDA error: %s", cudaGetErrorString(cudaGetLastError()));
    r->cuda_error = 1;
    free(r->h_results); free(r->h_errors); free(r->h_cycles); free(r);
    return NULL;

#else
    // --- CPU fallback path ---
    for (int i = 0; i < num_programs; i++) {
        int32_t gp[FLUX_NUM_REGS] = {0};
        int32_t stack[FLUX_STACK_SIZE];
        int sp = FLUX_STACK_SIZE;
        int pc = 0;
        int halted = 0;
        int cycles = 0;

        const uint8_t* bc = programs + offsets[i];
        uint32_t prog_len = lengths[i];

        while (!halted && pc >= 0 && (uint32_t)pc < prog_len && cycles < handle->max_cycles) {
            int result = cpu_flux_step(bc, prog_len, &pc, gp, stack, &sp, &halted);
            cycles++;
            if (result < 0) break;
            if (result > 0) pc += result;
            // result == 0: branch already set PC
        }

        r->h_results[i] = gp[0];
        r->h_errors[i]  = halted ? FLUX_ERR_NONE : FLUX_ERR_MAX_CYCLES;
        r->h_cycles[i]  = cycles;
    }
    r->gpu_ms     = 0.0f;
    r->cuda_error = 0;
    r->results    = 0;
    return r;
#endif
}

void flux_batch_free_result(FluxBatchHandle* handle, FluxBatchResult* result) {
    if (!result) return;
    free(result->h_results);
    free(result->h_errors);
    free(result->h_cycles);
    free(result);
}

// ============================================================================
// Query API
// ============================================================================

int flux_batch_device_count(void) {
#ifdef FLUX_CUDA_AVAILABLE
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
#else
    return 0;
#endif
}

int flux_batch_available(void) {
    return flux_batch_device_count() > 0 ? 1 : 0;
}

int flux_batch_device_info(int device_id, char* buf, int buf_len) {
#ifdef FLUX_CUDA_AVAILABLE
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        snprintf(buf, buf_len, "error: %s", cudaGetErrorString(err));
        return -1;
    }
    snprintf(buf, buf_len,
             "Device %d: %s (SM %d.%d, %d SMs, %d threads/SM, %.0f MHz, %.0f MB)",
             device_id, prop.name,
             prop.major, prop.minor,
             prop.multiProcessorCount,
             prop.maxThreadsPerMultiProcessor,
             prop.clockRate / 1000.0,
             prop.totalGlobalMem / (1024.0 * 1024.0));
    return 0;
#else
    snprintf(buf, buf_len, "CUDA not available (compiled without FLUX_CUDA_AVAILABLE)");
    return -1;
#endif
}

// ============================================================================
// Utility — Bytecode Packing
// ============================================================================

int flux_batch_pack(
    const uint8_t** programs,
    const int*      prog_lengths,
    int             num_programs,
    uint8_t**       out_packed,
    uint32_t**      out_offsets,
    uint32_t**      out_lengths,
    size_t*         out_total_size
) {
    if (!programs || !prog_lengths || num_programs <= 0) return -1;

    // Calculate total size with 4-byte alignment
    size_t total = 0;
    for (int i = 0; i < num_programs; i++) {
        total += prog_lengths[i];
        // Align to 4 bytes
        total = (total + 3) & ~3;
    }

    uint8_t*  packed  = (uint8_t*)calloc(total, 1);
    uint32_t* offsets = (uint32_t*)calloc(num_programs, sizeof(uint32_t));
    uint32_t* lengths = (uint32_t*)calloc(num_programs, sizeof(uint32_t));

    if (!packed || !offsets || !lengths) {
        free(packed); free(offsets); free(lengths);
        return -1;
    }

    size_t pos = 0;
    for (int i = 0; i < num_programs; i++) {
        offsets[i] = (uint32_t)pos;
        lengths[i] = (uint32_t)prog_lengths[i];
        memcpy(packed + pos, programs[i], prog_lengths[i]);
        pos += prog_lengths[i];
        // Align to 4 bytes
        pos = (pos + 3) & ~3;
    }

    *out_packed     = packed;
    *out_offsets    = offsets;
    *out_lengths    = lengths;
    *out_total_size = total;
    return 0;
}
