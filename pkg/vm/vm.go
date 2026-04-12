// Package vm implements the FLUX Unified ISA bytecode interpreter.
//
// This is the second runtime implementation in the fleet (alongside the
// Python VM in flux-runtime). Both implement the unified ISA where
// HALT=0x00 and A2A opcodes live at 0x50-0x5F.
//
// Register conventions:
//   R0: Hardwired zero (writes silently ignored)
//   R1-R7: General purpose
//   R8-R15: Float/frame pointers
//   R16-R31: Temporaries
//   R32-R63: Special/vector
package vm

import (
	"errors"
	"fmt"
)

// Sentinel errors for VM execution
var (
	ErrHalted         = errors.New("VM halted")
	ErrCycleLimit     = errors.New("cycle limit exceeded")
	ErrDivisionByZero = errors.New("division by zero")
	ErrStackOverflow  = errors.New("stack overflow")
	ErrStackUnderflow = errors.New("stack underflow")
	ErrStub           = errors.New("agent opcode stub (not implemented)")
	ErrInvalidOpcode  = errors.New("invalid opcode")
)

const (
	maxStackDepth  = 65536
	defaultMaxCycles = 10_000_000
)

// Flags holds CPU condition flags set by comparison operations.
type Flags struct {
	Zero     bool
	Sign     bool
	Carry    bool
	Overflow bool
}

// VM implements the FLUX Unified ISA bytecode interpreter.
type VM struct {
	Registers  [64]int32
	Memory     []byte
	Stack      []int32
	PC         int
	Halted     bool
	Cycles     uint64
	MaxCycles  uint64
	Flags      Flags
	Bytecode   []byte
}

// New creates a new VM with the given bytecode.
func New(bytecode []byte) *VM {
	return &VM{
		Bytecode:  bytecode,
		Memory:    make([]byte, 65536),
		Stack:     make([]int32, 0, 1024),
		MaxCycles: defaultMaxCycles,
	}
}

// Execute runs the bytecode until HALT, error, or cycle limit.
func (v *VM) Execute() error {
	for !v.Halted && v.Cycles < v.MaxCycles {
		if v.PC >= len(v.Bytecode) {
			v.Halted = true
			return nil
		}

		op := v.Bytecode[v.PC]
		v.Cycles++

		switch {
		case op == 0x00: // HALT
			v.Halted = true
			return nil

		case op == 0x01: // NOP
			v.PC++

		case op == 0x02: // RET
			if len(v.Stack) == 0 {
				return ErrStackUnderflow
			}
			v.PC = int(v.Stack[len(v.Stack)-1])
			v.Stack = v.Stack[:len(v.Stack)-1]

		// Format B: 2 bytes [op][rd]
		case op == 0x08: // INC rd
			rd := int(v.Bytecode[v.PC+1])
			if rd != 0 {
				v.Registers[rd]++
			}
			v.PC += 2

		case op == 0x09: // DEC rd
			rd := int(v.Bytecode[v.PC+1])
			if rd != 0 {
				v.Registers[rd]--
			}
			v.PC += 2

		case op == 0x0A: // NOT rd (bitwise)
			rd := int(v.Bytecode[v.PC+1])
			if rd != 0 {
				v.Registers[rd] = ^v.Registers[rd]
			}
			v.PC += 2

		case op == 0x0B: // NEG rd
			rd := int(v.Bytecode[v.PC+1])
			if rd != 0 {
				v.Registers[rd] = -v.Registers[rd]
			}
			v.PC += 2

		case op == 0x0C: // PUSH rd
			rd := int(v.Bytecode[v.PC+1])
			if len(v.Stack) >= maxStackDepth {
				return ErrStackOverflow
			}
			v.Stack = append(v.Stack, v.Registers[rd])
			v.PC += 2

		case op == 0x0D: // POP rd
			rd := int(v.Bytecode[v.PC+1])
			if len(v.Stack) == 0 {
				return ErrStackUnderflow
			}
			val := v.Stack[len(v.Stack)-1]
			v.Stack = v.Stack[:len(v.Stack)-1]
			if rd != 0 {
				v.Registers[rd] = val
			}
			v.PC += 2

		// Format F: 4 bytes [op][rd][imm16_lo][imm16_hi]
		case op == 0x18: // MOVI rd, imm16
			rd := int(v.Bytecode[v.PC+1])
			imm := int32(int16(uint16(v.Bytecode[v.PC+2]) | uint16(v.Bytecode[v.PC+3])<<8))
			if rd != 0 {
				v.Registers[rd] = imm
			}
			v.PC += 4

		case op == 0x19: // ADDI rd, imm16
			rd := int(v.Bytecode[v.PC+1])
			imm := int32(int16(uint16(v.Bytecode[v.PC+2]) | uint16(v.Bytecode[v.PC+3])<<8))
			if rd != 0 {
				v.Registers[rd] += imm
			}
			v.PC += 4

		case op == 0x1A: // SUBI rd, imm16
			rd := int(v.Bytecode[v.PC+1])
			imm := int32(int16(uint16(v.Bytecode[v.PC+2]) | uint16(v.Bytecode[v.PC+3])<<8))
			if rd != 0 {
				v.Registers[rd] -= imm
			}
			v.PC += 4

		// Format E: 4 bytes [op][rd][rs1][rs2]
		case op == 0x20: // ADD rd, rs1, rs2
			rd, rs1, rs2 := int(v.Bytecode[v.PC+1]), int(v.Bytecode[v.PC+2]), int(v.Bytecode[v.PC+3])
			if rd != 0 {
				v.Registers[rd] = v.Registers[rs1] + v.Registers[rs2]
			}
			v.PC += 4

		case op == 0x21: // SUB rd, rs1, rs2
			rd, rs1, rs2 := int(v.Bytecode[v.PC+1]), int(v.Bytecode[v.PC+2]), int(v.Bytecode[v.PC+3])
			if rd != 0 {
				v.Registers[rd] = v.Registers[rs1] - v.Registers[rs2]
			}
			v.PC += 4

		case op == 0x22: // MUL rd, rs1, rs2
			rd, rs1, rs2 := int(v.Bytecode[v.PC+1]), int(v.Bytecode[v.PC+2]), int(v.Bytecode[v.PC+3])
			if rd != 0 {
				v.Registers[rd] = v.Registers[rs1] * v.Registers[rs2]
			}
			v.PC += 4

		case op == 0x23: // DIV rd, rs1, rs2
			rd, rs1, rs2 := int(v.Bytecode[v.PC+1]), int(v.Bytecode[v.PC+2]), int(v.Bytecode[v.PC+3])
			if v.Registers[rs2] == 0 {
				return ErrDivisionByZero
			}
			if rd != 0 {
				// Truncate toward zero (Go semantics)
				a, b := v.Registers[rs1], v.Registers[rs2]
				v.Registers[rd] = a / b
			}
			v.PC += 4

		case op == 0x24: // MOD rd, rs1, rs2
			rd, rs1, rs2 := int(v.Bytecode[v.PC+1]), int(v.Bytecode[v.PC+2]), int(v.Bytecode[v.PC+3])
			if v.Registers[rs2] == 0 {
				return ErrDivisionByZero
			}
			if rd != 0 {
				v.Registers[rd] = v.Registers[rs1] % v.Registers[rs2]
			}
			v.PC += 4

		case op == 0x25: // AND rd, rs1, rs2
			rd, rs1, rs2 := int(v.Bytecode[v.PC+1]), int(v.Bytecode[v.PC+2]), int(v.Bytecode[v.PC+3])
			if rd != 0 {
				v.Registers[rd] = v.Registers[rs1] & v.Registers[rs2]
			}
			v.PC += 4

		case op == 0x26: // OR rd, rs1, rs2
			rd, rs1, rs2 := int(v.Bytecode[v.PC+1]), int(v.Bytecode[v.PC+2]), int(v.Bytecode[v.PC+3])
			if rd != 0 {
				v.Registers[rd] = v.Registers[rs1] | v.Registers[rs2]
			}
			v.PC += 4

		case op == 0x27: // XOR rd, rs1, rs2
			rd, rs1, rs2 := int(v.Bytecode[v.PC+1]), int(v.Bytecode[v.PC+2]), int(v.Bytecode[v.PC+3])
			if rd != 0 {
				v.Registers[rd] = v.Registers[rs1] ^ v.Registers[rs2]
			}
			v.PC += 4

		// Comparison: Format E, result 0/1 in rd, sets flags
		case op == 0x2C: // CMP_EQ rd, rs1, rs2
			rd, rs1, rs2 := int(v.Bytecode[v.PC+1]), int(v.Bytecode[v.PC+2]), int(v.Bytecode[v.PC+3])
			eq := v.Registers[rs1] == v.Registers[rs2]
			v.Flags.Zero = eq
			if rd != 0 {
				if eq {
					v.Registers[rd] = 1
				} else {
					v.Registers[rd] = 0
				}
			}
			v.PC += 4

		case op == 0x2D: // CMP_LT rd, rs1, rs2
			rd, rs1, rs2 := int(v.Bytecode[v.PC+1]), int(v.Bytecode[v.PC+2]), int(v.Bytecode[v.PC+3])
			lt := v.Registers[rs1] < v.Registers[rs2]
			v.Flags.Zero = lt
			if rd != 0 {
				if lt {
					v.Registers[rd] = 1
				} else {
					v.Registers[rd] = 0
				}
			}
			v.PC += 4

		case op == 0x2E: // CMP_GT rd, rs1, rs2
			rd, rs1, rs2 := int(v.Bytecode[v.PC+1]), int(v.Bytecode[v.PC+2]), int(v.Bytecode[v.PC+3])
			gt := v.Registers[rs1] > v.Registers[rs2]
			v.Flags.Zero = gt
			if rd != 0 {
				if gt {
					v.Registers[rd] = 1
				} else {
					v.Registers[rd] = 0
				}
			}
			v.PC += 4

		case op == 0x2F: // CMP_NE rd, rs1, rs2
			rd, rs1, rs2 := int(v.Bytecode[v.PC+1]), int(v.Bytecode[v.PC+2]), int(v.Bytecode[v.PC+3])
			ne := v.Registers[rs1] != v.Registers[rs2]
			v.Flags.Zero = ne
			if rd != 0 {
				if ne {
					v.Registers[rd] = 1
				} else {
					v.Registers[rd] = 0
				}
			}
			v.PC += 4

		// Branch: Format F with signed imm16 offset
		case op == 0x43: // JMP offset
			imm := int16(uint16(v.Bytecode[v.PC+2]) | uint16(v.Bytecode[v.PC+3])<<8)
			v.PC += 4
			v.PC += int(imm)

		case op == 0x44: // JZ rd, offset (jump if rd == 0)
			rd := int(v.Bytecode[v.PC+1])
			imm := int16(uint16(v.Bytecode[v.PC+2]) | uint16(v.Bytecode[v.PC+3])<<8)
			v.PC += 4
			if v.Registers[rd] == 0 {
				v.PC += int(imm)
			}

		case op == 0x45: // JNZ rd, offset (jump if rd != 0)
			rd := int(v.Bytecode[v.PC+1])
			imm := int16(uint16(v.Bytecode[v.PC+2]) | uint16(v.Bytecode[v.PC+3])<<8)
			v.PC += 4
			if v.Registers[rd] != 0 {
				v.PC += int(imm)
			}

		case op == 0x4A: // CALL offset (push return addr, jump)
			imm := int16(uint16(v.Bytecode[v.PC+2]) | uint16(v.Bytecode[v.PC+3])<<8)
			retAddr := v.PC + 4
			v.Stack = append(v.Stack, int32(retAddr))
			v.PC = retAddr + int(imm)

		// Agent opcodes (stubs)
		case op == 0x50: // TELL
			return ErrStub
		case op == 0x51: // ASK
			return ErrStub
		case op == 0x53: // BCAST
			return ErrStub

		default:
			return fmt.Errorf("%w: 0x%02x at PC=%d", ErrInvalidOpcode, op, v.PC)
		}
	}

	if v.Cycles >= v.MaxCycles {
		return ErrCycleLimit
	}
	return nil
}

// Encoding helpers for building bytecode programmatically.

// EncodeMOVI encodes: MOVI rd, imm16 (4 bytes)
func EncodeMOVI(rd byte, imm int16) []byte {
	return []byte{0x18, rd, byte(imm), byte(imm >> 8)}
}

// EncodeE encodes a Format E instruction: [op][rd][rs1][rs2]
func EncodeE(op, rd, rs1, rs2 byte) []byte {
	return []byte{op, rd, rs1, rs2}
}

// EncodeBranch encodes a branch: [op][rd][offset_lo][offset_hi]
func EncodeBranch(op, rd byte, offset int16) []byte {
	return []byte{op, rd, byte(offset), byte(offset >> 8)}
}

// EncodeJMP encodes: JMP offset (rd=0)
func EncodeJMP(offset int16) []byte {
	return EncodeBranch(0x43, 0, offset)
}

// EncodeCALL encodes: CALL offset (rd=0)
func EncodeCALL(offset int16) []byte {
	return EncodeBranch(0x4A, 0, offset)
}

// EncodeB encodes a Format B instruction: [op][rd]
func EncodeB(op, rd byte) []byte {
	return []byte{op, rd}
}
