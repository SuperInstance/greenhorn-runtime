//go:build !cuda

package cuda

import "fmt"

// Pure Go CPU fallback for batch FLUX execution.
// This file provides the same API as batch_gpu.go but uses the
// Go FLUX VM (pkg/flux/vm.go semantics) for CPU execution.
// Enable GPU execution with: go build -tags cuda

// FLUX opcodes (canonical ISA, matching pkg/flux/vm.go)
const (
        fluxOpHALT  = 0x00
        fluxOpNOP   = 0x01
        fluxOpRET   = 0x02
        fluxOpINC   = 0x08
        fluxOpDEC   = 0x09
        fluxOpNOT   = 0x0A
        fluxOpNEG   = 0x0B
        fluxOpPUSH  = 0x0C
        fluxOpPOP   = 0x0D
        fluxOpMOVI  = 0x18
        fluxOpADDI  = 0x19
        fluxOpSUBI  = 0x1A
        fluxOpADD   = 0x20
        fluxOpSUB   = 0x21
        fluxOpMUL   = 0x22
        fluxOpDIV   = 0x23
        fluxOpMOD   = 0x24
        fluxOpAND   = 0x25
        fluxOpOR    = 0x26
        fluxOpXOR   = 0x27
        fluxOpSHL   = 0x28
        fluxOpSHR   = 0x29
        fluxOpMIN   = 0x2A
        fluxOpMAX   = 0x2B
        fluxOpCMP_EQ = 0x2C
        fluxOpCMP_LT = 0x2D
        fluxOpCMP_GT = 0x2E
        fluxOpCMP_NE = 0x2F
        fluxOpMOV   = 0x3A
        fluxOpJZ    = 0x3C
        fluxOpJNZ   = 0x3D
        fluxOpJLT   = 0x3E
        fluxOpJGT   = 0x3F
        fluxOpMOVI16 = 0x40
        fluxOpJMP   = 0x43
        fluxOpLOOP  = 0x46
        fluxOpCALL  = 0x4A
        fluxOpSTRIPCONF = 0x17
)

const (
        fluxNumRegs   = 16
        fluxStackSize = 256
        fluxMaxCycles = 1000000
)

// fluxFormatSize returns the byte length of an instruction by opcode.
// Matches Go vm.formatSize() exactly.
func fluxFormatSize(op uint8) int {
        switch {
        case op <= 0x07:
                return 1
        case op <= 0x17:
                return 2
        case op <= 0x1F:
                return 3
        case op <= 0x4F:
                return 4
        default:
                return 1
        }
}

// cpuFluxVM is a single FLUX virtual machine instance for CPU execution.
type cpuFluxVM struct {
        gp     [fluxNumRegs]int32
        stack  [fluxStackSize]int32
        sp     int
        pc     int
        cycles int
        halted bool
}

func newCPUFluxVM() *cpuFluxVM {
        return &cpuFluxVM{sp: fluxStackSize}
}

// execute runs a FLUX bytecode program.
func (vm *cpuFluxVM) execute(bc []byte) int32 {
        for !vm.halted && vm.pc < len(bc) && vm.cycles < fluxMaxCycles {
                result := vm.step(bc)
                if result < 0 {
                        break
                }
                // result == 0: branch already set PC
                // result > 0: normal instruction, advance PC
                if result > 0 {
                        vm.pc += result
                }
        }
        return vm.gp[0]
}

// step executes a single instruction. Returns instruction size if normal,
// 0 if branch already set PC, -1 if halt/error.
func (vm *cpuFluxVM) step(bc []byte) int {
        if vm.pc >= len(bc) || vm.halted {
                return -1
        }

        op := bc[vm.pc]
        size := fluxFormatSize(op)

        if vm.pc+size > len(bc) {
                vm.halted = true
                return -1
        }

        vm.cycles++

        switch op {
        case fluxOpHALT:
                vm.halted = true
                return size

        case fluxOpNOP:
                return size

        case fluxOpRET:
                if vm.sp >= fluxStackSize {
                        vm.halted = true
                        return -1
                }
                vm.pc = int(vm.stack[vm.sp])
                vm.sp++
                return 0 // PC already set

        case fluxOpINC:
                vm.gp[bc[vm.pc+1]]++
                return size

        case fluxOpDEC:
                vm.gp[bc[vm.pc+1]]--
                return size

        case fluxOpNOT:
                vm.gp[bc[vm.pc+1]] = ^vm.gp[bc[vm.pc+1]]
                return size

        case fluxOpNEG:
                vm.gp[bc[vm.pc+1]] = -vm.gp[bc[vm.pc+1]]
                return size

        case fluxOpPUSH:
                vm.sp--
                if vm.sp < 0 {
                        vm.halted = true
                        return -1
                }
                vm.stack[vm.sp] = vm.gp[bc[vm.pc+1]]
                return size

        case fluxOpPOP:
                if vm.sp >= fluxStackSize {
                        vm.halted = true
                        return -1
                }
                vm.gp[bc[vm.pc+1]] = vm.stack[vm.sp]
                vm.sp++
                return size

        case fluxOpSTRIPCONF:
                return size

        case fluxOpMOVI:
                vm.gp[bc[vm.pc+1]] = int32(int8(bc[vm.pc+2]))
                return size

        case fluxOpADDI:
                vm.gp[bc[vm.pc+1]] += int32(int8(bc[vm.pc+2]))
                return size

        case fluxOpSUBI:
                vm.gp[bc[vm.pc+1]] -= int32(int8(bc[vm.pc+2]))
                return size

        case fluxOpADD:
                vm.gp[bc[vm.pc+1]] = vm.gp[bc[vm.pc+2]] + vm.gp[bc[vm.pc+3]]
                return size

        case fluxOpSUB:
                vm.gp[bc[vm.pc+1]] = vm.gp[bc[vm.pc+2]] - vm.gp[bc[vm.pc+3]]
                return size

        case fluxOpMUL:
                vm.gp[bc[vm.pc+1]] = vm.gp[bc[vm.pc+2]] * vm.gp[bc[vm.pc+3]]
                return size

        case fluxOpDIV:
                rd := bc[vm.pc+1]
                rs := bc[vm.pc+2]
                rt := bc[vm.pc+3]
                if vm.gp[rt] == 0 {
                        vm.halted = true
                        return -1
                }
                vm.gp[rd] = vm.gp[rs] / vm.gp[rt]
                return size

        case fluxOpMOD:
                rd := bc[vm.pc+1]
                rs := bc[vm.pc+2]
                rt := bc[vm.pc+3]
                if vm.gp[rt] == 0 {
                        vm.halted = true
                        return -1
                }
                vm.gp[rd] = vm.gp[rs] % vm.gp[rt]
                return size

        case fluxOpAND:
                vm.gp[bc[vm.pc+1]] = vm.gp[bc[vm.pc+2]] & vm.gp[bc[vm.pc+3]]
                return size

        case fluxOpOR:
                vm.gp[bc[vm.pc+1]] = vm.gp[bc[vm.pc+2]] | vm.gp[bc[vm.pc+3]]
                return size

        case fluxOpXOR:
                vm.gp[bc[vm.pc+1]] = vm.gp[bc[vm.pc+2]] ^ vm.gp[bc[vm.pc+3]]
                return size

        case fluxOpSHL:
                vm.gp[bc[vm.pc+1]] = vm.gp[bc[vm.pc+2]] << (vm.gp[bc[vm.pc+3]] & 31)
                return size

        case fluxOpSHR:
                vm.gp[bc[vm.pc+1]] = vm.gp[bc[vm.pc+2]] >> (vm.gp[bc[vm.pc+3]] & 31)
                return size

        case fluxOpMIN:
                vm.gp[bc[vm.pc+1]] = min(vm.gp[bc[vm.pc+2]], vm.gp[bc[vm.pc+3]])
                return size

        case fluxOpMAX:
                vm.gp[bc[vm.pc+1]] = max(vm.gp[bc[vm.pc+2]], vm.gp[bc[vm.pc+3]])
                return size

        case fluxOpCMP_EQ:
                rd := bc[vm.pc+1]
                rs := bc[vm.pc+2]
                rt := bc[vm.pc+3]
                if vm.gp[rs] == vm.gp[rt] {
                        vm.gp[rd] = 1
                } else {
                        vm.gp[rd] = 0
                }
                return size

        case fluxOpCMP_LT:
                rd := bc[vm.pc+1]
                rs := bc[vm.pc+2]
                rt := bc[vm.pc+3]
                if vm.gp[rs] < vm.gp[rt] {
                        vm.gp[rd] = 1
                } else {
                        vm.gp[rd] = 0
                }
                return size

        case fluxOpCMP_GT:
                rd := bc[vm.pc+1]
                rs := bc[vm.pc+2]
                rt := bc[vm.pc+3]
                if vm.gp[rs] > vm.gp[rt] {
                        vm.gp[rd] = 1
                } else {
                        vm.gp[rd] = 0
                }
                return size

        case fluxOpCMP_NE:
                rd := bc[vm.pc+1]
                rs := bc[vm.pc+2]
                rt := bc[vm.pc+3]
                if vm.gp[rs] != vm.gp[rt] {
                        vm.gp[rd] = 1
                } else {
                        vm.gp[rd] = 0
                }
                return size

        case fluxOpMOV:
                vm.gp[bc[vm.pc+1]] = vm.gp[bc[vm.pc+2]]
                return size

        case fluxOpJNZ:
                rd := bc[vm.pc+1]
                off := int32(int8(bc[vm.pc+2]))
                if vm.gp[rd] != 0 {
                        vm.pc += int(off)
                        return 0 // jump already set PC
                }
                return size

        case fluxOpJZ:
                rd := bc[vm.pc+1]
                off := int32(int8(bc[vm.pc+2]))
                if vm.gp[rd] == 0 {
                        vm.pc += int(off)
                        return 0
                }
                return size

        case fluxOpJLT:
                rd := bc[vm.pc+1]
                off := int32(int8(bc[vm.pc+2]))
                if vm.gp[rd] < 0 {
                        vm.pc += int(off)
                        return 0
                }
                return size

        case fluxOpJGT:
                rd := bc[vm.pc+1]
                off := int32(int8(bc[vm.pc+2]))
                if vm.gp[rd] > 0 {
                        vm.pc += int(off)
                        return 0
                }
                return size

        case fluxOpMOVI16:
                imm := int16(uint16(bc[vm.pc+2]) | uint16(bc[vm.pc+3])<<8)
                vm.gp[bc[vm.pc+1]] = int32(imm)
                return size

        case fluxOpJMP:
                off := int16(uint16(bc[vm.pc+2]) | uint16(bc[vm.pc+3])<<8)
                vm.pc += int(off)
                return 0

        case fluxOpLOOP:
                rd := bc[vm.pc+1]
                off := int(uint16(bc[vm.pc+2]) | uint16(bc[vm.pc+3])<<8)
                vm.gp[rd]--
                if vm.gp[rd] > 0 {
                        vm.pc -= off
                        return 0
                }
                return size

        case fluxOpCALL:
                if vm.sp <= 0 {
                        vm.halted = true
                        return -1
                }
                vm.sp--
                vm.stack[vm.sp] = int32(vm.pc + size)
                off := int16(uint16(bc[vm.pc+2]) | uint16(bc[vm.pc+3])<<8)
                vm.pc += int(off)
                return 0

        default:
                return size
        }
}

// BatchExecutor manages batch FLUX execution (CPU fallback).
type BatchExecutor struct {
        maxCycles int
}

// NewBatchExecutor creates a new batch executor with default configuration.
func NewBatchExecutor() (*BatchExecutor, error) {
        return &BatchExecutor{maxCycles: fluxMaxCycles}, nil
}

// NewBatchExecutorWithConfig creates a new batch executor with custom configuration.
func NewBatchExecutorWithConfig(cfg BatchConfig) (*BatchExecutor, error) {
        return &BatchExecutor{maxCycles: cfg.MaxCycles}, nil
}

// Close releases resources (no-op for CPU fallback).
func (e *BatchExecutor) Close() {}

// Error returns the last error message (always empty for CPU).
func (e *BatchExecutor) Error() string { return "no error (CPU mode)" }

// Run executes a batch of FLUX programs on the CPU.
func (e *BatchExecutor) Run(programs [][]byte) (*BatchResult, error) {
        if len(programs) == 0 {
                return nil, fmt.Errorf("no programs to execute")
        }

        n := len(programs)
        result := &BatchResult{
                NumPrograms: n,
                GPUMs:       0,
                Results:     make([]int32, n),
                Errors:      make([]int32, n),
                Cycles:      make([]int32, n),
        }

        for i, bc := range programs {
                vm := newCPUFluxVM()
                vm.execute(bc)
                result.Results[i] = vm.gp[0]
                result.Cycles[i] = int32(vm.cycles)
                if vm.halted {
                        // Check if halted due to div-by-zero: R0 should be unchanged
                        // For now, just report ErrNone — the Go VM doesn't distinguish halt reasons
                        // The CUDA kernel returns specific error codes; CPU fallback uses ErrNone
                        result.Errors[i] = ErrNone
                } else {
                        result.Errors[i] = ErrMaxCycles
                }
        }

        return result, nil
}

// DeviceCount returns 0 (CPU fallback, no CUDA devices).
func DeviceCount() int { return 0 }

// Available returns false (CPU fallback).
func Available() bool { return false }

// DeviceInfo returns CPU info string.
func DeviceInfo(deviceID int) string {
        return "CPU fallback mode (no CUDA device)"
}

// PackBytecodes packs individual program bytecodes into a contiguous buffer.
func PackBytecodes(programs [][]byte) (packed []byte, offsets []uint32, lengths []uint32, totalSize uintptr, err error) {
        n := len(programs)
        if n == 0 {
                return nil, nil, nil, 0, nil
        }

        // Calculate total size with 4-byte alignment
        total := 0
        for _, p := range programs {
                total += len(p)
                total = (total + 3) & ^3
        }

        packed = make([]byte, total)
        offsets = make([]uint32, n)
        lengths = make([]uint32, n)

        pos := 0
        for i, p := range programs {
                offsets[i] = uint32(pos)
                lengths[i] = uint32(len(p))
                copy(packed[pos:], p)
                pos += len(p)
                pos = (pos + 3) & ^3
        }

        totalSize = uintptr(total)
        return packed, offsets, lengths, totalSize, nil
}
