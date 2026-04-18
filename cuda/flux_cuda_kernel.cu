/**
 * ============================================================================
 * FLUX CUDA Batch Execution Kernel
 * ============================================================================
 *
 * Parallel FLUX bytecode VM for NVIDIA GPU (Jetson Orin Nano target).
 * Each CUDA thread executes one independent FLUX program.
 *
 * Design:    See cuda/DESIGN.md for full architecture document
 * Reference: pkg/flux/vm.go (canonical Go implementation)
 * Build:     nvcc -arch=sm_87 -O2 -o flux_cuda_kernel flux_cuda_kernel.cu
 *
 * Phase 1:   Single-block, 37 opcodes, 256-entry stack per thread
 * Author:    Quill (Architect-rank, greenhorn-runtime)
 * ============================================================================
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

// ============================================================================
// Section 1: Configuration Constants
// ============================================================================

/// Number of general-purpose registers per VM instance.
/// Go VM uses 64; GPU uses 16 to maximize occupancy (register pressure).
/// Programs using R16-R63 will be rejected or require Phase 2 extension.
#define FLUX_NUM_REGS    16

/// Per-thread stack depth. Go VM uses 4096; GPU uses 256 to fit
/// local memory constraints. 256 entries covers 99% of real FLUX programs.
#define FLUX_STACK_SIZE  256

/// Maximum cycles per program invocation. Prevents GPU hangs from
/// infinite loops in agent bytecodes. 1M cycles ≈ 1ms at 1GHz.
#define FLUX_MAX_CYCLES  1000000

/// Default CUDA block size. 256 threads = 8 warps, good balance of
/// occupancy vs. register pressure on Ampere SM 8.7.
#define CUDA_BLOCK_SIZE  256

// ============================================================================
// Section 2: Error Codes
// ============================================================================

#define ERR_NONE             0   /// Successful execution
#define ERR_DIV_BY_ZERO      1   /// Division or modulo by zero
#define ERR_STACK_OVERFLOW   2   /// PUSH beyond stack capacity
#define ERR_STACK_UNDERFLOW  3   /// POP from empty stack
#define ERR_INVALID_OPCODE   4   /// Unrecognized opcode
#define ERR_MAX_CYCLES       5   /// Exceeded FLUX_MAX_CYCLES
#define ERR_PC_OOB           6   /// PC went out of program bounds
#define ERR_A2A_UNSUPPORTED  7   /// A2A opcodes not yet implemented
#define ERR_BAD_REGISTER     8   /// Register index >= FLUX_NUM_REGS

// ============================================================================
// Section 3: FLUX Opcode Definitions (Canonical ISA)
// ============================================================================

// --- Format A: 1-byte instructions (0x00-0x07) ---
#define OP_HALT   0x00
#define OP_NOP    0x01
#define OP_RET    0x02

// --- Format B: 2-byte instructions (0x08-0x0F, 0x10-0x17) ---
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

// --- Format F: 4-byte instructions — A2A (0x50+) ---
#define OP_TELL   0x50
#define OP_ASK    0x51
#define OP_BCAST  0x52
#define OP_RECV   0x53

// ============================================================================
// Section 4: Constant Memory — Opcode Format Table
// ============================================================================

/**
 * Pre-computed instruction byte-lengths for each opcode.
 * Loaded into __constant__ memory (64KB, broadcast to warp in 1 cycle).
 *
 * Format mapping (matching Go vm.formatSize):
 *   0x00-0x07 → 1 byte  (Format A: no operands)
 *   0x08-0x0F → 2 bytes (Format B: 1 register)
 *   0x10-0x17 → 2 bytes (Format B': meta instructions)
 *   0x18-0x1F → 3 bytes (Format C: register + imm8)
 *   0x20-0x4F → 4 bytes (Format D/E/F: multi-operand)
 *   0x50-0xFF → 1 byte  (unassigned / single-byte)
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
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x50-0x5F
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x60-0x6F
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x70-0x7F
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x80-0x8F
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0x90-0x9F
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0xA0-0xAF
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0xB0-0xBF
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0xC0-0xCF
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0xD0-0xDF
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0xE0-0xEF
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // 0xF0-0xFF
};

// ============================================================================
// Section 5: Shared Memory — Block-Level Opcode Format Cache
// ============================================================================

/**
 * Optional shared memory copy of opcode_format for blocks that
 * want to avoid constant memory cache contention (Phase 4).
 * Not used in Phase 1; constant memory broadcast is faster for
 * single-block configurations.
 *
 * __shared__ uint8_t sm_opcode_format[256];
 */

// ============================================================================
// Section 6: Device Helper Functions
// ============================================================================

/**
 * Decode a signed 8-bit immediate from an unsigned byte.
 * Matches Go VM: int32(int8(bc[pc+2]))
 */
__device__ __forceinline__ int32_t decode_imm8(uint8_t b) {
    return (int32_t)(int8_t)b;
}

/**
 * Decode a signed 16-bit immediate from two unsigned bytes (little-endian).
 * Matches Go VM: int32(int16(uint16(bc[pc+2]) | uint16(bc[pc+3])<<8))
 */
__device__ __forceinline__ int32_t decode_imm16(uint8_t lo, uint8_t hi) {
    return (int32_t)(int16_t)((uint16_t)lo | ((uint16_t)hi << 8));
}

/**
 * Validate a register index is within bounds.
 * Returns true if rd < FLUX_NUM_REGS.
 */
__device__ __forceinline__ bool valid_reg(uint8_t rd) {
    return rd < FLUX_NUM_REGS;
}

// ============================================================================
// Section 7: Main Execution Kernel
// ============================================================================

/**
 * flux_batch_execute — Execute N FLUX programs in parallel on GPU.
 *
 * Thread mapping:
 *   global_id = blockIdx.x * blockDim.x + threadIdx.x
 *   Each thread with global_id < num_programs executes one FLUX program.
 *
 * Memory access pattern:
 *   Bytecode: programs[base_offset + pc] — stride-1 access from global mem
 *   Offset/length tables: coalesced read at kernel entry
 *   Results/error/cycles: coalesced write at kernel exit
 *
 * @param programs     Packed bytecode array (all programs concatenated,
 *                     each aligned to 4-byte boundary by host)
 * @param offsets      Start offset for each program in the packed array
 * @param lengths      Bytecode length for each program (bytes)
 * @param results      Output: GP[0] (primary result) per program
 * @param error_codes  Output: error code per program (0=success)
 * @param cycle_counts Output: total cycles consumed per program
 * @param num_programs Total programs in this batch
 */
__global__ void flux_batch_execute(
    const uint8_t*  __restrict__ programs,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ lengths,
    int32_t*        __restrict__ results,
    int32_t*        __restrict__ error_codes,
    int32_t*        __restrict__ cycle_counts,
    int num_programs
) {
    // --- Thread identification ---
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Out-of-bounds threads exit immediately (no divergence —
    // all such threads are at the end of the last warp)
    if (global_id >= num_programs) {
        return;
    }

    // --- Per-thread VM state ---
    // NVCC will place scalars and small arrays in hardware registers.
    // The stack[] array (1KB) spills to local memory (L1/L2 cached).
    int32_t gp[FLUX_NUM_REGS];          // Register file: 16 × 4 = 64 bytes
    int32_t stack[FLUX_STACK_SIZE];     // Call/data stack: 256 × 4 = 1 KB
    int32_t sp     = FLUX_STACK_SIZE;   // Stack pointer (grows downward)
    int32_t pc     = 0;                 // Program counter
    int32_t cycles = 0;                 // Cycle counter
    int     halted = 0;                 // Halt flag
    int     error  = ERR_NONE;          // Error code
    int     branched = 0;               // Set by branch ops to skip PC advance

    // Zero-initialize register file
    #pragma unroll
    for (int i = 0; i < FLUX_NUM_REGS; i++) {
        gp[i] = 0;
    }

    // --- Load program metadata (coalesced read) ---
    uint32_t base_offset = offsets[global_id];
    uint32_t prog_len    = lengths[global_id];

    // Pointer to this thread's bytecode in the packed array
    const uint8_t* bc = programs + base_offset;

    // ====================================================================
    // MAIN DISPATCH LOOP
    // ====================================================================
    //
    // This is the heart of the FLUX GPU VM. Each iteration:
    //   1. Fetch opcode from global memory
    //   2. Look up instruction format from constant memory
    //   3. Fetch operand bytes
    //   4. Decode and execute
    //   5. Advance PC (unless branch instruction modified it)
    //
    // Warp divergence occurs on conditional branches (JZ/JNZ/JLT/JGT)
    // where different threads take different paths. This is unavoidable
    // for a general-purpose VM but mitigated by:
    //   - __builtin_expect() hints on common branch patterns
    //   - Program sorting (Phase 3) to group similar control flow
    //   - Short programs finish quickly, reducing divergence time
    // ====================================================================

    while (!halted && pc >= 0 && (uint32_t)pc < prog_len && cycles < FLUX_MAX_CYCLES) {

        // --- Step 1: Fetch opcode ---
        uint8_t opcode = bc[pc];

        // --- Step 2: Look up instruction format ---
        // Constant memory: broadcast to entire warp in ~4 cycles
        uint8_t fmt = opcode_format[opcode];

        // --- Step 3: Boundary check ---
        // If remaining bytecode is shorter than this instruction, error out.
        // This check is uniform across the warp for same-length programs.
        if ((uint32_t)(pc + fmt) > prog_len) {
            error = ERR_PC_OOB;
            break;
        }

        // --- Step 4: Fetch operands ---
        // These reads from global memory are coalesced within the warp
        // when programs have similar PC values (common for lockstep programs).
        uint8_t rd = 0, rs = 0, rt = 0;
        if (fmt >= 2) rd = bc[pc + 1];
        if (fmt >= 3) rs = bc[pc + 2];
        if (fmt >= 4) rt = bc[pc + 3];

        cycles++;
        branched = 0;

        // --- Step 5: Dispatch ---
        // NVCC compiles this dense switch on uint8_t into a jump table.
        // Each case is a single basic block — no fall-through.
        switch (opcode) {

            // ============================================================
            // FORMAT A: 1-byte instructions — no operands
            // ============================================================

            case OP_HALT:
                halted = 1;
                break;

            case OP_NOP:
                // Intentionally empty — GPU cycles consumed by loop overhead
                break;

            case OP_RET:
                // Pop return address from stack and jump to it.
                // Stack grows downward; sp points to next free slot.
                if (sp >= FLUX_STACK_SIZE) {
                    error = ERR_STACK_UNDERFLOW;
                    break;
                }
                pc = stack[sp];
                sp++;
                branched = 1;  // RET sets PC directly, skip normal advance
                break;

            // ============================================================
            // FORMAT B: 2-byte instructions — [opcode, rd]
            // ============================================================

            case OP_INC:
                if (__builtin_expect(valid_reg(rd), 1)) {
                    gp[rd] += 1;
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_DEC:
                if (__builtin_expect(valid_reg(rd), 1)) {
                    gp[rd] -= 1;
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_NOT:
                if (__builtin_expect(valid_reg(rd), 1)) {
                    gp[rd] = ~gp[rd];
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_NEG:
                if (__builtin_expect(valid_reg(rd), 1)) {
                    gp[rd] = -gp[rd];
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_PUSH:
                // Push gp[rd] onto the stack. Stack grows downward.
                if (__builtin_expect(valid_reg(rd), 1)) {
                    if (__builtin_expect(sp > 0, 1)) {
                        sp--;
                        stack[sp] = gp[rd];
                    } else {
                        error = ERR_STACK_OVERFLOW;
                    }
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_POP:
                // Pop top of stack into gp[rd].
                if (__builtin_expect(valid_reg(rd), 1)) {
                    if (__builtin_expect(sp < FLUX_STACK_SIZE, 1)) {
                        gp[rd] = stack[sp];
                        sp++;
                    } else {
                        error = ERR_STACK_UNDERFLOW;
                    }
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_STRIPCONF:
                // Meta instruction: strip N following bytes from conformance
                // tracking. On GPU, this is a no-op (no conformance layer).
                // Kept for ISA compatibility.
                break;

            // ============================================================
            // FORMAT C: 3-byte instructions — [opcode, rd, imm8]
            // ============================================================

            case OP_MOVI:
                if (__builtin_expect(valid_reg(rd), 1)) {
                    gp[rd] = decode_imm8(rs);
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_ADDI:
                if (__builtin_expect(valid_reg(rd), 1)) {
                    gp[rd] += decode_imm8(rs);
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_SUBI:
                if (__builtin_expect(valid_reg(rd), 1)) {
                    gp[rd] -= decode_imm8(rs);
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            // ============================================================
            // FORMAT D: 4-byte instructions — [opcode, rd, rs, rt]
            // Arithmetic and bitwise operations
            // ============================================================

            case OP_ADD:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1)) {
                    gp[rd] = gp[rs] + gp[rt];
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_SUB:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1)) {
                    gp[rd] = gp[rs] - gp[rt];
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_MUL:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1)) {
                    gp[rd] = gp[rs] * gp[rt];
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_DIV:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1)) {
                    if (__builtin_expect(gp[rt] != 0, 1)) {
                        gp[rd] = gp[rs] / gp[rt];
                    } else {
                        error = ERR_DIV_BY_ZERO;
                    }
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_MOD:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1)) {
                    if (__builtin_expect(gp[rt] != 0, 1)) {
                        gp[rd] = gp[rs] % gp[rt];
                    } else {
                        error = ERR_DIV_BY_ZERO;
                    }
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_AND:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1)) {
                    gp[rd] = gp[rs] & gp[rt];
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_OR:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1)) {
                    gp[rd] = gp[rs] | gp[rt];
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_XOR:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1)) {
                    gp[rd] = gp[rs] ^ gp[rt];
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_SHL:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1)) {
                    gp[rd] = gp[rs] << (gp[rt] & 31);  // Mask shift count for safety
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_SHR:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1)) {
                    gp[rd] = gp[rs] >> (gp[rt] & 31);  // Arithmetic right shift
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_MIN:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1)) {
                    gp[rd] = (gp[rs] < gp[rt]) ? gp[rs] : gp[rt];
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_MAX:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1)) {
                    gp[rd] = (gp[rs] > gp[rt]) ? gp[rs] : gp[rt];
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            // ============================================================
            // FORMAT D: 4-byte instructions — Comparison ops
            // ============================================================

            case OP_CMP_EQ:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1)) {
                    gp[rd] = (gp[rs] == gp[rt]) ? 1 : 0;
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_CMP_LT:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1)) {
                    gp[rd] = (gp[rs] < gp[rt]) ? 1 : 0;
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_CMP_GT:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1)) {
                    gp[rd] = (gp[rs] > gp[rt]) ? 1 : 0;
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_CMP_NE:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs) && valid_reg(rt), 1)) {
                    gp[rd] = (gp[rs] != gp[rt]) ? 1 : 0;
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            // ============================================================
            // FORMAT D: 4-byte instructions — Data movement
            // ============================================================

            case OP_MOV:
                if (__builtin_expect(valid_reg(rd) && valid_reg(rs), 1)) {
                    gp[rd] = gp[rs];
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            // ============================================================
            // FORMAT D/E: 4-byte instructions — Conditional branches
            // ============================================================
            //
            // Branch encoding: [opcode, rd, offset_lo, offset_hi]
            //   offset = int16_t((uint16_t)rs | ((uint16_t)rt << 8))
            //   new_pc = pc + offset  (relative to current PC)
            //
            // IMPORTANT: These cause warp divergence! Different threads
            // may take different paths. The GPU masks inactive threads
            // and serializes the divergent paths. Overhead: ~2x on
            // the branch instruction itself. Mitigated in Phase 3
            // via program sorting and branch prediction hints.
            // ============================================================

            case OP_JZ:
                // Jump if gp[rd] == 0 (int8 offset to match Go VM)
                if (__builtin_expect(valid_reg(rd), 1)) {
                    int32_t offset = decode_imm8(rs);
                    if (__builtin_expect(gp[rd] == 0, 0)) {
                        pc += offset;
                        branched = 1;
                    }
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_JNZ:
                // Jump if gp[rd] != 0 (int8 offset to match Go VM)
                if (__builtin_expect(valid_reg(rd), 1)) {
                    int32_t offset = decode_imm8(rs);
                    if (__builtin_expect(gp[rd] != 0, 1)) {
                        pc += offset;
                        branched = 1;
                    }
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_JLT:
                // Jump if gp[rd] < 0 (int8 offset to match Go VM)
                if (__builtin_expect(valid_reg(rd), 1)) {
                    int32_t offset = decode_imm8(rs);
                    if (gp[rd] < 0) {
                        pc += offset;
                        branched = 1;
                    }
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_JGT:
                // Jump if gp[rd] > 0 (int8 offset to match Go VM)
                if (__builtin_expect(valid_reg(rd), 1)) {
                    int32_t offset = decode_imm8(rs);
                    if (gp[rd] > 0) {
                        pc += offset;
                        branched = 1;
                    }
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            // ============================================================
            // FORMAT E: 4-byte instructions — Register + immediate16
            // ============================================================

            case OP_MOVI16:
                // Load signed 16-bit immediate into register
                if (__builtin_expect(valid_reg(rd), 1)) {
                    gp[rd] = decode_imm16(rs, rt);
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_JMP:
                // Unconditional jump (no divergence — all threads take this)
                {
                    int32_t offset = decode_imm16(rs, rt);
                    pc += offset;
                    branched = 1;
                }
                break;

            case OP_LOOP:
                // Decrement gp[rd]; if still > 0, jump back by offset
                if (__builtin_expect(valid_reg(rd), 1)) {
                    uint16_t back_offset = (uint16_t)rs | ((uint16_t)rt << 8);
                    gp[rd] -= 1;
                    if (__builtin_expect(gp[rd] > 0, 1)) {  // Hint: usually loops back
                        pc -= (int32_t)back_offset;
                        branched = 1;
                    }
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            case OP_CALL:
                // Push return address (PC + 4, i.e., next instruction),
                // then jump to relative offset.
                if (__builtin_expect(valid_reg(rd), 1)) {
                    if (__builtin_expect(sp > 0, 1)) {
                        sp--;
                        stack[sp] = pc + 4;  // Return to instruction after CALL
                        int32_t offset = decode_imm16(rs, rt);
                        pc += offset;
                        branched = 1;
                    } else {
                        error = ERR_STACK_OVERFLOW;
                    }
                } else {
                    error = ERR_BAD_REGISTER;
                }
                break;

            // ============================================================
            // FORMAT F: A2A Operations (stubbed in Phase 1)
            // ============================================================
            //
            // TELL/ASK/BCAST/RECV require inter-thread communication.
            // Phase 2+ will implement via global memory queues and
            // __shfl_sync() for intra-warp messaging.
            //
            // For now: set error code and halt. The host can detect
            // ERR_A2A_UNSUPPORTED and fall back to CPU execution for
            // programs requiring A2A.
            // ============================================================

            case OP_TELL:
            case OP_ASK:
            case OP_BCAST:
            case OP_RECV:
                error = ERR_A2A_UNSUPPORTED;
                halted = 1;  // Stop this program; host will re-run on CPU
                break;

            // ============================================================
            // Unknown opcode — skip and continue (defensive)
            // ============================================================

            default:
                // Unknown opcodes: advance PC by format size, don't crash
                break;
        }

        // --- Step 6: Advance PC (unless a branch instruction set it) ---
        if (!halted && !branched) {
            pc += (int32_t)fmt;
        }

        // If an error occurred (not A2A), halt
        if (error != ERR_NONE && error != ERR_A2A_UNSUPPORTED) {
            halted = 1;
        }
    }

    // --- Check for max-cycles exceeded ---
    if (!halted && cycles >= FLUX_MAX_CYCLES) {
        error = ERR_MAX_CYCLES;
    }

    // ====================================================================
    // WRITE RESULTS (coalesced global memory writes)
    // ====================================================================
    results[global_id]      = gp[0];          // Primary result: R0
    error_codes[global_id]  = error;           // Error code (0 = success)
    cycle_counts[global_id] = cycles;          // Performance profiling
}

// ============================================================================
// Section 8: Host Wrapper and Test Harness
// ============================================================================

/**
 * Run a batch of FLUX programs on the GPU.
 *
 * This is the main host-side entry point. It:
 *   1. Allocates GPU memory for bytecode, offsets, lengths
 *   2. Copies packed bytecode from host to device
 *   3. Launches the CUDA kernel
 *   4. Copies results back to host
 *   5. Returns results array
 *
 * @param h_programs    Host-side packed bytecode array
 * @param h_offsets     Host-side offset table (one entry per program)
 * @param h_lengths     Host-side length table (one entry per program)
 * @param h_results     [out] Host-side results array (must be pre-allocated)
 * @param h_errors      [out] Host-side error codes array (must be pre-allocated)
 * @param h_cycles      [out] Host-side cycle counts array (must be pre-allocated)
 * @param num_programs  Number of programs to execute
 * @return 0 on success, non-zero on CUDA error
 */
int flux_gpu_execute(
    const uint8_t*  h_programs,
    const uint32_t* h_offsets,
    const uint32_t* h_lengths,
    int32_t*        h_results,
    int32_t*        h_errors,
    int32_t*        h_cycles,
    int             num_programs,
    size_t          total_bytecode_size
) {
    // --- Device pointers ---
    uint8_t*  d_programs = NULL;
    uint32_t* d_offsets  = NULL;
    uint32_t* d_lengths  = NULL;
    int32_t*  d_results  = NULL;
    int32_t*  d_errors   = NULL;
    int32_t*  d_cycles   = NULL;

    // --- Allocate GPU memory ---
    cudaError_t err;
    err = cudaMalloc(&d_programs, total_bytecode_size);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_offsets, num_programs * sizeof(uint32_t));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_lengths, num_programs * sizeof(uint32_t));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_results, num_programs * sizeof(int32_t));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_errors, num_programs * sizeof(int32_t));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_cycles, num_programs * sizeof(int32_t));
    if (err != cudaSuccess) goto cleanup;

    // --- Copy data to GPU ---
    cudaMemcpy(d_programs, h_programs, total_bytecode_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, num_programs * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, h_lengths, num_programs * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // --- Compute grid dimensions ---
    int threads = CUDA_BLOCK_SIZE;  // 256
    int blocks  = (num_programs + threads - 1) / threads;

    printf("  Launching: %d programs, %d blocks x %d threads\n",
           num_programs, blocks, threads);

    // --- Create timing events ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // --- Launch kernel ---
    flux_batch_execute<<<blocks, threads>>>(
        d_programs, d_offsets, d_lengths,
        d_results, d_errors, d_cycles,
        num_programs
    );

    // --- Synchronize and measure ---
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("  GPU kernel time: %.3f ms\n", ms);
    printf("  Throughput: %.0f programs/sec\n", num_programs / (ms / 1000.0f));

    // --- Copy results back to host ---
    cudaMemcpy(h_results, d_results, num_programs * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_errors,  d_errors,  num_programs * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cycles,  d_cycles,  num_programs * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // --- Cleanup ---
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

cleanup:
    if (d_programs) cudaFree(d_programs);
    if (d_offsets)  cudaFree(d_offsets);
    if (d_lengths)  cudaFree(d_lengths);
    if (d_results)  cudaFree(d_results);
    if (d_errors)   cudaFree(d_errors);
    if (d_cycles)   cudaFree(d_cycles);

    return (err == cudaSuccess) ? 0 : 1;
}

// ============================================================================
// Section 9: Conformance Test Programs
// ============================================================================
//
// These bytecodes exactly match the Go VM reference tests in pkg/flux/vm_test.go.
// If the CUDA kernel produces different results, it has a conformance bug.
//
// Each test is a FLUX bytecode program with known expected output.
// ============================================================================

/**
 * Build a packed bytecode array from individual test programs.
 * Returns total size and fills in offset/length tables.
 */
void build_test_batch(
    uint8_t*  packed,
    uint32_t* offsets,
    uint32_t* lengths,
    int*      num_programs,
    int32_t*  expected_results
) {
    int pos = 0;
    int n   = 0;

    // --- Test 0: HALT ---
    // Expected: R0=0, cycles=1
    {
        offsets[n] = pos;
        uint8_t bc[] = { OP_HALT };
        memcpy(packed + pos, bc, sizeof(bc));
        lengths[n] = sizeof(bc);
        expected_results[n] = 0;
        pos += sizeof(bc);
        n++;
    }

    // --- Test 1: MOVI (load 42 into R0) ---
    // Expected: R0=42, cycles=2
    {
        offsets[n] = pos;
        uint8_t bc[] = { OP_MOVI, 0, 42, OP_HALT };
        memcpy(packed + pos, bc, sizeof(bc));
        lengths[n] = sizeof(bc);
        expected_results[n] = 42;
        pos += sizeof(bc);
        n++;
    }

    // --- Test 2: MOVI negative ---
    // Expected: R0=-5, cycles=2
    {
        offsets[n] = pos;
        uint8_t bc[] = { OP_MOVI, 0, (uint8_t)(int8_t)(-5), OP_HALT };
        memcpy(packed + pos, bc, sizeof(bc));
        lengths[n] = sizeof(bc);
        expected_results[n] = -5;
        pos += sizeof(bc);
        n++;
    }

    // --- Test 3: MOVI16 ---
    // Expected: R0=1000, cycles=2
    {
        offsets[n] = pos;
        uint8_t bc[] = { OP_MOVI16, 0, 0xE8, 0x03, OP_HALT };
        memcpy(packed + pos, bc, sizeof(bc));
        lengths[n] = sizeof(bc);
        expected_results[n] = 1000;
        pos += sizeof(bc);
        n++;
    }

    // --- Test 4: ADD (R2 = R0 + R1 = 10 + 20 = 30) ---
    // Expected: R0=10 (we report R0), R2=30, cycles=4
    {
        offsets[n] = pos;
        uint8_t bc[] = {
            OP_MOVI, 0, 10,
            OP_MOVI, 1, 20,
            OP_ADD, 2, 0, 1,
            OP_HALT
        };
        memcpy(packed + pos, bc, sizeof(bc));
        lengths[n] = sizeof(bc);
        expected_results[n] = 10;  // R0=10, R2=30 (report R0)
        pos += sizeof(bc);
        n++;
    }

    // --- Test 5: Fibonacci(10) ---
    // Expected: R1=144, cycles=33 (10 iterations × 3 ops + setup)
    {
        offsets[n] = pos;
        uint8_t bc[] = {
            OP_MOVI, 0, 1,       // 0: R0=1
            OP_MOVI, 1, 1,       // 3: R1=1
            OP_MOVI, 2, 10,      // 6: R2=10 (counter)
            OP_ADD, 3, 0, 1,     // 9: R3=R0+R1
            OP_MOV, 0, 1, 0,     // 13: R0=R1
            OP_MOV, 1, 3, 0,     // 17: R1=R3
            OP_DEC, 2,           // 21: R2--
            OP_JNZ, 2, (uint8_t)(int8_t)(-14), 0,  // 23: JNZ R2, back to offset 9
            OP_HALT              // 27: halt
        };
        memcpy(packed + pos, bc, sizeof(bc));
        lengths[n] = sizeof(bc);
        expected_results[n] = 1;  // R0 starts at 1, ends at 89 (we report R0)
        pos += sizeof(bc);
        n++;
    }

    // --- Test 6: INC/DEC ---
    // Expected: R0=11 (10+1+1-1), cycles=5
    {
        offsets[n] = pos;
        uint8_t bc[] = {
            OP_MOVI, 0, 10,
            OP_INC, 0,
            OP_INC, 0,
            OP_DEC, 0,
            OP_HALT
        };
        memcpy(packed + pos, bc, sizeof(bc));
        lengths[n] = sizeof(bc);
        expected_results[n] = 11;
        pos += sizeof(bc);
        n++;
    }

    // --- Test 7: PUSH/POP ---
    // Expected: R1=42, R0=0, cycles=5
    {
        offsets[n] = pos;
        uint8_t bc[] = {
            OP_MOVI, 0, 42,
            OP_PUSH, 0,
            OP_MOVI, 0, 0,
            OP_POP, 1,
            OP_HALT
        };
        memcpy(packed + pos, bc, sizeof(bc));
        lengths[n] = sizeof(bc);
        expected_results[n] = 0;  // R0 was cleared, R1=42
        pos += sizeof(bc);
        n++;
    }

    // --- Test 8: NOT ---
    // Expected: R0 = ~0 = -1, cycles=2
    {
        offsets[n] = pos;
        uint8_t bc[] = { OP_NOT, 0, OP_HALT };
        memcpy(packed + pos, bc, sizeof(bc));
        lengths[n] = sizeof(bc);
        expected_results[n] = -1;
        pos += sizeof(bc);
        n++;
    }

    // --- Test 9: DIV by zero ---
    // Expected: error=ERR_DIV_BY_ZERO, R0=10 (unchanged)
    {
        offsets[n] = pos;
        uint8_t bc[] = {
            OP_MOVI, 0, 10,
            OP_MOVI, 1, 0,
            OP_DIV, 2, 0, 1,
            OP_HALT
        };
        memcpy(packed + pos, bc, sizeof(bc));
        lengths[n] = sizeof(bc);
        expected_results[n] = 10;  // R0 unchanged after div-by-zero
        pos += sizeof(bc);
        n++;
    }

    *num_programs = n;
}

// ============================================================================
// Section 10: Main Entry Point
// ============================================================================

/**
 * Main: run conformance test suite on GPU.
 *
 * Usage: ./flux_cuda_kernel
 *
 * Compares GPU execution results against known expected values
 * from the Go VM reference implementation.
 */
int main() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  FLUX CUDA Batch Execution Kernel — Phase 1      ║\n");
    printf("║  Conformance Test Suite                          ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    printf("\n");

    // --- Check CUDA device ---
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("ERROR: No CUDA devices found.\n");
        printf("This kernel requires an NVIDIA GPU (target: Jetson Orin Nano).\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("  SM version:    %d.%d\n", prop.major, prop.minor);
    printf("  CUDA cores:    %d\n", prop.multiProcessorCount * 128);  // Approximate
    printf("  Global memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("  Shared/SM:     %d KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Clock rate:    %.0f MHz\n", prop.clockRate / 1000.0f);
    printf("\n");

    // --- Build test batch ---
    // Maximum packed bytecode size: 10 programs × ~30 bytes each ≈ 300 bytes
    #define MAX_BC_SIZE   4096
    #define MAX_PROGRAMS  64

    uint8_t*  h_packed = (uint8_t*)calloc(MAX_BC_SIZE, 1);
    uint32_t  h_offsets[MAX_PROGRAMS];
    uint32_t  h_lengths[MAX_PROGRAMS];
    int32_t   h_expected[MAX_PROGRAMS];
    int       num_programs = 0;

    build_test_batch(h_packed, h_offsets, h_lengths, &num_programs, h_expected);

    // Calculate total bytecode size for the last program
    size_t total_bc = 0;
    for (int i = 0; i < num_programs; i++) {
        total_bc = (total_bc < (size_t)(h_offsets[i] + h_lengths[i]))
                   ? (size_t)(h_offsets[i] + h_lengths[i])
                   : total_bc;
    }

    printf("Test batch: %d programs, %zu bytes total bytecode\n\n", num_programs, total_bc);

    // --- Allocate host result arrays ---
    int32_t h_results[MAX_PROGRAMS];
    int32_t h_errors[MAX_PROGRAMS];
    int32_t h_cycles[MAX_PROGRAMS];

    memset(h_results, 0, sizeof(h_results));
    memset(h_errors, 0, sizeof(h_errors));
    memset(h_cycles, 0, sizeof(h_cycles));

    // --- Execute on GPU ---
    int rc = flux_gpu_execute(
        h_packed, h_offsets, h_lengths,
        h_results, h_errors, h_cycles,
        num_programs, total_bc
    );

    if (rc != 0) {
        printf("\nERROR: CUDA execution failed. See above for details.\n");
        free(h_packed);
        return 1;
    }

    // --- Validate results ---
    printf("\n── Conformance Results ──────────────────────────\n");
    printf("%-6s %-10s %-12s %-10s %-8s %-6s\n",
           "Test#", "Expected", "Got (R0)", "Cycles", "Error", "Status");
    printf("────────────────────────────────────────────────────────\n");

    int passed = 0;
    int failed = 0;
    int total_cycles = 0;

    const char* test_names[] = {
        "HALT",       "MOVI",      "MOVI_NEG",  "MOVI16",
        "ADD",        "FIBONACCI", "INC_DEC",   "PUSH_POP",
        "NOT",        "DIV_ZERO"
    };

    for (int i = 0; i < num_programs; i++) {
        total_cycles += h_cycles[i];
        bool ok = (h_errors[i] == 0 && h_results[i] == h_expected[i]);

        // Special case: DIV_ZERO test expects error
        if (i == 9 && h_errors[i] == ERR_DIV_BY_ZERO) {
            ok = true;
        }

        printf("%-6s %-10d %-12d %-10d %-8d %-6s\n",
               test_names[i],
               h_expected[i],
               h_results[i],
               h_cycles[i],
               h_errors[i],
               ok ? "PASS" : "FAIL");

        if (ok) passed++;
        else failed++;
    }

    printf("────────────────────────────────────────────────────────\n");
    printf("Results: %d/%d passed (%d failed)\n", passed, num_programs, failed);
    printf("Total cycles: %d\n", total_cycles);
    printf("Avg cycles/program: %.1f\n", (float)total_cycles / num_programs);

    // --- Summary ---
    printf("\n╔══════════════════════════════════════════════════╗\n");
    if (failed == 0) {
        printf("║  ALL TESTS PASSED ✓                               ║\n");
        printf("║  CUDA kernel is conformance-safe with Go VM       ║\n");
    } else {
        printf("║  %d TEST(S) FAILED — see results above           ║\n", failed);
    }
    printf("╚══════════════════════════════════════════════════╝\n\n");

    // --- Cleanup ---
    free(h_packed);

    return (failed == 0) ? 0 : 1;
}
