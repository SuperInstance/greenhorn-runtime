/**
 * ============================================================================
 * FLUX Batch Executor — Host-Side GPU Memory Manager
 * ============================================================================
 *
 * C-callable API for managing GPU memory and launching batch FLUX kernels.
 * Designed for integration with Go runtime via CGo (batch.go).
 *
 * Usage:
 *   FluxBatchHandle h = flux_batch_init();
 *   FluxBatchResult* r = flux_batch_execute(h, programs, offsets, lengths, n, total_size);
 *   // use r->results[i], r->errors[i], r->cycles[i]
 *   flux_batch_free_result(h, r);
 *   flux_batch_destroy(h);
 *
 * Thread safety: Each FluxBatchHandle manages its own CUDA stream.
 * Multiple handles can be used concurrently from different OS threads.
 *
 * Reference: pkg/flux/vm.go (canonical Go FLUX VM)
 * Design:    cuda/DESIGN.md (architecture document)
 * ============================================================================
 */

#ifndef FLUX_BATCH_EXECUTOR_H
#define FLUX_BATCH_EXECUTOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// ============================================================================
// Error Codes (match kernel-side definitions)
// ============================================================================

#define FLUX_ERR_NONE             0
#define FLUX_ERR_DIV_BY_ZERO      1
#define FLUX_ERR_STACK_OVERFLOW   2
#define FLUX_ERR_STACK_UNDERFLOW  3
#define FLUX_ERR_INVALID_OPCODE   4
#define FLUX_ERR_MAX_CYCLES       5
#define FLUX_ERR_PC_OOB           6
#define FLUX_ERR_A2A_UNSUPPORTED  7
#define FLUX_ERR_BAD_REGISTER     8

// ============================================================================
// Configuration
// ============================================================================

/// Default CUDA block size. 256 threads = 8 warps, optimal for Ampere SM 8.7.
#define FLUX_CUDA_BLOCK_SIZE  256

/// Per-thread register count (GPU-optimized; Go VM uses 64).
#define FLUX_NUM_REGS         16

/// Per-thread stack depth (Go VM uses 4096; GPU uses 256 for occupancy).
#define FLUX_STACK_SIZE       256

/// Maximum cycles per program to prevent GPU hangs.
#define FLUX_MAX_CYCLES       1000000

// ============================================================================
// Data Structures
// ============================================================================

/**
 * FluxBatchHandle — Opaque handle to a GPU batch executor context.
 *
 * Owns:
 *   - CUDA stream for async kernel launches
 *   - Persistent GPU memory buffers (avoids per-launch malloc)
 *   - Last error message string
 */
typedef struct FluxBatchHandle FluxBatchHandle;

/**
 * FluxBatchResult — Output from a batch execution.
 *
 * All arrays are heap-allocated with `num_programs` entries.
 * Must be freed via flux_batch_free_result().
 */
typedef struct {
    int32_t  results;       ///< Device-side pointer to GP[0] per program
    int32_t* h_results;     ///< Host-side copy of results
    int32_t* h_errors;      ///< Host-side copy of error codes
    int32_t* h_cycles;      ///< Host-side copy of cycle counts
    int      num_programs;  ///< Number of programs executed
    float    gpu_ms;        ///< GPU kernel execution time in milliseconds
    int      cuda_error;    ///< 0 = success, non-zero = CUDA error code
    char     error_msg[256]; ///< Human-readable error message (if cuda_error != 0)
} FluxBatchResult;

/**
 * FluxBatchConfig — Configuration for the batch executor.
 *
 * Pass to flux_batch_init_ex() for custom settings.
 */
typedef struct {
    int block_size;      ///< CUDA threads per block (default: 256)
    int max_cycles;      ///< Max cycles per program (default: 1,000,000)
    int device_id;       ///< GPU device ID (default: 0)
} FluxBatchConfig;

// ============================================================================
// Lifecycle API
// ============================================================================

/**
 * Initialize a batch executor with default settings.
 * Uses GPU 0, block size 256, max cycles 1M.
 *
 * @return Handle on success, NULL on failure (no CUDA device, OOM, etc.)
 */
FluxBatchHandle* flux_batch_init(void);

/**
 * Initialize a batch executor with custom configuration.
 *
 * @param config  Configuration struct
 * @return Handle on success, NULL on failure
 */
FluxBatchHandle* flux_batch_init_ex(const FluxBatchConfig* config);

/**
 * Destroy a batch executor and free all GPU resources.
 *
 * @param handle  Executor to destroy (safe to pass NULL)
 */
void flux_batch_destroy(FluxBatchHandle* handle);

/**
 * Get last error message from the executor.
 *
 * @param handle  Executor handle
 * @return Static error string (valid until next call on same handle)
 */
const char* flux_batch_get_error(FluxBatchHandle* handle);

// ============================================================================
// Execution API
// ============================================================================

/**
 * Execute a batch of FLUX programs on the GPU.
 *
 * This is the primary entry point. It:
 *   1. Packs programs into GPU-friendly format (if needed)
 *   2. Uploads bytecode to GPU
 *   3. Launches the CUDA kernel
 *   4. Downloads results
 *   5. Returns a FluxBatchResult with per-program outputs
 *
 * @param handle           Batch executor handle
 * @param programs         Packed bytecode array (all programs concatenated)
 * @param offsets          Start offset for each program in packed array
 * @param lengths          Bytecode length for each program
 * @param num_programs     Number of programs to execute
 * @param total_bc_size    Total size of packed bytecode array in bytes
 * @return Result struct on success, NULL on failure
 */
FluxBatchResult* flux_batch_run(
    FluxBatchHandle*      handle,
    const uint8_t*        programs,
    const uint32_t*       offsets,
    const uint32_t*       lengths,
    int                   num_programs,
    size_t                total_bc_size
);

/**
 * Free a FluxBatchResult returned by flux_batch_run().
 *
 * @param handle  Executor handle (used to free device memory)
 * @param result  Result to free (safe to pass NULL)
 */
void flux_batch_free_result(FluxBatchHandle* handle, FluxBatchResult* result);

// ============================================================================
// Query API
// ============================================================================

/**
 * Get the number of available CUDA devices.
 *
 * @return Device count (0 if no CUDA-capable GPU)
 */
int flux_batch_device_count(void);

/**
 * Get GPU device properties as a formatted string.
 *
 * @param device_id  GPU device index
 * @param buf        Output buffer (must be >= 512 bytes)
 * @param buf_len    Buffer length
 * @return 0 on success, -1 on error
 */
int flux_batch_device_info(int device_id, char* buf, int buf_len);

/**
 * Check if the batch executor can run on this system.
 *
 * @return 1 if CUDA is available, 0 otherwise
 */
int flux_batch_available(void);

// ============================================================================
// Utility API — Bytecode Packing (for Go callers)
// ============================================================================

/**
 * Build packed bytecode arrays from individual program bytecodes.
 *
 * Go callers can use this to prepare data for flux_batch_run():
 *   1. Collect individual program bytecodes
 *   2. Call flux_batch_pack() to build the packed format
 *   3. Call flux_batch_run() with the packed arrays
 *
 * @param programs      Array of pointers to individual program bytecodes
 * @param prog_lengths  Length of each individual program
 * @param num_programs  Number of programs
 * @param out_packed    [out] Allocated packed bytecode buffer (caller must free)
 * @param out_offsets   [out] Allocated offsets array (caller must free)
 * @param out_lengths   [out] Allocated lengths array (caller must free)
 * @param out_total_size [out] Total packed bytecode size
 * @return 0 on success, -1 on error
 */
int flux_batch_pack(
    const uint8_t** programs,
    const int*      prog_lengths,
    int             num_programs,
    uint8_t**       out_packed,
    uint32_t**      out_offsets,
    uint32_t**      out_lengths,
    size_t*         out_total_size
);

#ifdef __cplusplus
}
#endif

#endif // FLUX_BATCH_EXECUTOR_H
