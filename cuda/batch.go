// Package cuda provides Go bindings for the FLUX CUDA batch execution engine.
//
// This package wraps the CUDA kernel (batch_kernel.cu) via CGo, enabling
// Go programs to execute batches of FLUX bytecodes on NVIDIA GPUs.
//
// When CUDA is not available, the package falls back to a CPU reference
// implementation that produces identical results, enabling testing on
// any machine.
//
// To build with CUDA support:
//
//	go build -tags cuda ./cuda/
//
// Without the cuda tag, a pure Go CPU implementation is used.
//
// # Usage
//
//	executor, err := cuda.NewBatchExecutor()
//	if err != nil { /* no GPU or other error */ }
//	defer executor.Close()
//
//	programs := [][]byte{
//	        {0x18, 0x00, 0x2A, 0x00}, // MOVI R0, 42; HALT
//	        {0x18, 0x00, 0x0A, 0x00}, // MOVI R0, 10; HALT
//	}
//	result, err := executor.Run(programs)
//	if err != nil { /* execution error */ }
//	defer result.Close()
//
//	fmt.Printf("R0[0] = %d\n", result.Results[0]) // 42
//	fmt.Printf("R0[1] = %d\n", result.Results[1]) // 10
//
// Reference: pkg/flux/vm.go (canonical Go FLUX VM)
// Design:    cuda/DESIGN.md (architecture document)
package cuda

import "fmt"

// ============================================================================
// Error Codes
// ============================================================================

// Error codes returned by the FLUX batch executor.
const (
	ErrNone           = 0
	ErrDivByZero      = 1
	ErrStackOverflow  = 2
	ErrStackUnderflow = 3
	ErrInvalidOpcode  = 4
	ErrMaxCycles      = 5
	ErrPCOutOfBounds  = 6
	ErrA2AUnsupported = 7
	ErrBadRegister    = 8
)

// ErrorString returns a human-readable description for a FLUX error code.
func ErrorString(code int) string {
	switch code {
	case ErrNone:
		return "success"
	case ErrDivByZero:
		return "division by zero"
	case ErrStackOverflow:
		return "stack overflow"
	case ErrStackUnderflow:
		return "stack underflow"
	case ErrInvalidOpcode:
		return "invalid opcode"
	case ErrMaxCycles:
		return "max cycles exceeded"
	case ErrPCOutOfBounds:
		return "PC out of bounds"
	case ErrA2AUnsupported:
		return "A2A operation not supported"
	case ErrBadRegister:
		return "bad register index"
	default:
		return fmt.Sprintf("unknown error code %d", code)
	}
}

// ============================================================================
// Configuration
// ============================================================================

// BatchConfig holds configuration options for the batch executor.
type BatchConfig struct {
	// BlockSize is the number of CUDA threads per block (default: 256).
	BlockSize int

	// MaxCycles is the maximum cycles per program before halting (default: 1,000,000).
	MaxCycles int

	// DeviceID is the GPU device index to use (default: 0).
	DeviceID int
}

// DefaultConfig returns a BatchConfig with sensible defaults.
func DefaultConfig() BatchConfig {
	return BatchConfig{
		BlockSize: 256,
		MaxCycles: 1000000,
		DeviceID:  0,
	}
}

// ============================================================================
// BatchResult
// ============================================================================

// BatchResult holds the output from a batch execution.
type BatchResult struct {
	// Results contains GP[0] (the primary result register) for each program.
	Results []int32

	// Errors contains the error code for each program (0 = success).
	Errors []int32

	// Cycles contains the total cycles consumed by each program.
	Cycles []int32

	// GPUMs is the GPU kernel execution time in milliseconds (0 for CPU fallback).
	GPUMs float32

	// NumPrograms is the number of programs that were executed.
	NumPrograms int
}

// Close frees resources associated with the result.
func (r *BatchResult) Close() {}

// AllOK returns true if all programs executed without errors.
func (r *BatchResult) AllOK() bool {
	for _, e := range r.Errors {
		if e != ErrNone {
			return false
		}
	}
	return true
}

// ErrorCount returns the number of programs that had errors.
func (r *BatchResult) ErrorCount() int {
	count := 0
	for _, e := range r.Errors {
		if e != ErrNone {
			count++
		}
	}
	return count
}
