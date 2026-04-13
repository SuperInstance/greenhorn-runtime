//go:build cuda

package cuda

/*
#cgo CFLAGS: -I${SRCDIR} -DFLUX_CUDA_AVAILABLE
#cgo linux LDFLAGS: -lcudart -lm

#include "batch_executor.cuh"

#ifdef __cplusplus
extern "C" {
#endif

FluxBatchHandle* flux_batch_init(void);
FluxBatchHandle* flux_batch_init_ex(const FluxBatchConfig* config);
void flux_batch_destroy(FluxBatchHandle* handle);
const char* flux_batch_get_error(FluxBatchHandle* handle);
FluxBatchResult* flux_batch_run(
    FluxBatchHandle* handle,
    const unsigned char* programs,
    const unsigned int* offsets,
    const unsigned int* lengths,
    int num_programs,
    size_t total_bc_size
);
void flux_batch_free_result(FluxBatchHandle* handle, FluxBatchResult* result);
int flux_batch_device_count(void);
int flux_batch_available(void);
int flux_batch_device_info(int device_id, char* buf, int buf_len);
int flux_batch_pack(
    const unsigned char** programs,
    const int* prog_lengths,
    int num_programs,
    unsigned char** out_packed,
    unsigned int** out_offsets,
    unsigned int** out_lengths,
    size_t* out_total_size
);

#ifdef __cplusplus
}
#endif
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// BatchExecutor manages a CUDA context for batch FLUX execution.
// Each executor owns its own CUDA stream and GPU memory buffers.
// Use Close() to release resources.
type BatchExecutor struct {
	handle *C.FluxBatchHandle
}

// NewBatchExecutor creates a new batch executor with default configuration.
func NewBatchExecutor() (*BatchExecutor, error) {
	h := C.flux_batch_init()
	if h == nil {
		return nil, fmt.Errorf("failed to init batch executor (no CUDA device?)")
	}
	e := &BatchExecutor{handle: h}
	runtime.SetFinalizer(e, (*BatchExecutor).Close)
	return e, nil
}

// NewBatchExecutorWithConfig creates a new batch executor with custom configuration.
func NewBatchExecutorWithConfig(cfg BatchConfig) (*BatchExecutor, error) {
	cCfg := C.FluxBatchConfig{
		block_size: C.int(cfg.BlockSize),
		max_cycles: C.int(cfg.MaxCycles),
		device_id:  C.int(cfg.DeviceID),
	}
	h := C.flux_batch_init_ex(&cCfg)
	if h == nil {
		return nil, fmt.Errorf("failed to init batch executor with config")
	}
	e := &BatchExecutor{handle: h}
	runtime.SetFinalizer(e, (*BatchExecutor).Close)
	return e, nil
}

// Close releases all GPU resources held by the executor.
func (e *BatchExecutor) Close() {
	if e.handle != nil {
		C.flux_batch_destroy(e.handle)
		e.handle = nil
	}
	runtime.SetFinalizer(e, nil)
}

// Error returns the last error message from the executor.
func (e *BatchExecutor) Error() string {
	if e.handle == nil {
		return "executor is closed"
	}
	return C.GoString(C.flux_batch_get_error(e.handle))
}

// Run executes a batch of FLUX programs on the GPU.
//
// Each element of `programs` is a FLUX bytecode array. All programs are
// packed into a single GPU buffer and executed in parallel (one CUDA thread
// per program).
//
// Returns a BatchResult with per-program results, error codes, and cycle counts.
func (e *BatchExecutor) Run(programs [][]byte) (*BatchResult, error) {
	if e.handle == nil {
		return nil, fmt.Errorf("executor is closed")
	}
	if len(programs) == 0 {
		return nil, fmt.Errorf("no programs to execute")
	}

	n := len(programs)

	// Pack programs into contiguous buffer
	packed, offsets, lengths, totalSize, err := packBytecodes(programs)
	if err != nil {
		return nil, fmt.Errorf("pack bytecodes: %w", err)
	}
	defer freePacked(packed, offsets, lengths)

	// Execute on GPU
	var cPacked unsafe.Pointer
	if len(packed) > 0 {
		cPacked = unsafe.Pointer(&packed[0])
	}
	var cOffsets unsafe.Pointer
	if len(offsets) > 0 {
		cOffsets = unsafe.Pointer(&offsets[0])
	}
	var cLengths unsafe.Pointer
	if len(lengths) > 0 {
		cLengths = unsafe.Pointer(&lengths[0])
	}

	cResult := C.flux_batch_run(
		e.handle,
		(*C.uchar)(cPacked),
		(*C.uint)(cOffsets),
		(*C.uint)(cLengths),
		C.int(n),
		C.size_t(totalSize),
	)
	if cResult == nil {
		return nil, fmt.Errorf("batch execution failed: %s", e.Error())
	}
	defer C.flux_batch_free_result(e.handle, cResult)

	// Copy results to Go slices
	result := &BatchResult{
		NumPrograms: n,
		GPUMs:       float32(cResult.gpu_ms),
		Results:     make([]int32, n),
		Errors:      make([]int32, n),
		Cycles:      make([]int32, n),
	}

	cResults := unsafe.Slice(cResult.h_results, n)
	cErrors := unsafe.Slice(cResult.h_errors, n)
	cCycles := unsafe.Slice(cResult.h_cycles, n)

	copy(result.Results, cResults)
	copy(result.Errors, cErrors)
	copy(result.Cycles, cCycles)

	return result, nil
}

// DeviceCount returns the number of CUDA-capable GPUs available.
func DeviceCount() int {
	return int(C.flux_batch_device_count())
}

// Available returns true if CUDA is available on this system.
func Available() bool {
	return C.flux_batch_available() != 0
}

// DeviceInfo returns a human-readable string describing the GPU.
func DeviceInfo(deviceID int) string {
	buf := make([]byte, 512)
	C.flux_batch_device_info(C.int(deviceID), (*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf)))
	end := 0
	for end < len(buf) && buf[end] != 0 {
		end++
	}
	return string(buf[:end])
}

// PackBytecodes packs individual program bytecodes into a contiguous buffer.
func PackBytecodes(programs [][]byte) (packed []byte, offsets []uint32, lengths []uint32, totalSize uintptr, err error) {
	return packBytecodes(programs)
}

func packBytecodes(programs [][]byte) (packed []byte, offsets []uint32, lengths []uint32, totalSize uintptr, err error) {
	n := len(programs)

	cProgs := make([]*C.uchar, n)
	cLens := make([]C.int, n)
	for i, p := range programs {
		if len(p) > 0 {
			cProgs[i] = (*C.uchar)(unsafe.Pointer(&p[0]))
		}
		cLens[i] = C.int(len(p))
	}

	var cPacked *C.uchar
	var cOffsets *C.uint
	var cLengths *C.uint
	var cTotalSize C.size_t

	ret := C.flux_batch_pack(
		(**C.uchar)(unsafe.Pointer(&cProgs[0])),
		(*C.int)(unsafe.Pointer(&cLens[0])),
		C.int(n),
		&cPacked,
		&cOffsets,
		&cLengths,
		&cTotalSize,
	)

	if ret != 0 || cPacked == nil {
		return nil, nil, nil, 0, fmt.Errorf("failed to pack bytecodes")
	}

	totalSize = uintptr(cTotalSize)
	if totalSize > 0 {
		packed = unsafe.Slice(cPacked, totalSize)
	}
	offsets = unsafe.Slice(cOffsets, n)
	lengths = unsafe.Slice(cLengths, n)

	return packed, offsets, lengths, totalSize, nil
}

func freePacked(packed []byte, offsets []uint32, lengths []uint32) {
	if len(packed) > 0 {
		C.free(unsafe.Pointer(&packed[0]))
	}
	if len(offsets) > 0 {
		C.free(unsafe.Pointer(&offsets[0]))
	}
	if len(lengths) > 0 {
		C.free(unsafe.Pointer(&lengths[0]))
	}
}
