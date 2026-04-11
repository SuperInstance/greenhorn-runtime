package flux

const (
	OpHALT = 0x00
	OpNOP  = 0x01
	OpRET  = 0x02
	OpINC  = 0x08
	OpDEC  = 0x09
	OpNOT  = 0x0A
	OpNEG  = 0x0B
	OpPUSH = 0x0C
	OpPOP  = 0x0D
	OpSTRIPCONF = 0x17
	OpMOVI = 0x18
	OpADDI = 0x19
	OpSUBI = 0x1A
	OpADD  = 0x20
	OpSUB  = 0x21
	OpMUL  = 0x22
	OpDIV  = 0x23
	OpMOD  = 0x24
	OpAND  = 0x25
	OpOR   = 0x26
	OpXOR  = 0x27
	OpSHL  = 0x28
	OpSHR  = 0x29
	OpMIN  = 0x2A
	OpMAX  = 0x2B
	OpCMP_EQ = 0x2C
	OpCMP_LT = 0x2D
	OpCMP_GT = 0x2E
	OpCMP_NE = 0x2F
	OpMOV  = 0x3A
	OpJZ   = 0x3C
	OpJNZ  = 0x3D
	OpJLT  = 0x3E
	OpJGT  = 0x3F
	OpMOVI16 = 0x40
	OpJMP  = 0x43
	OpLOOP = 0x46
	NumRegs = 64
)

type VM struct {
	GP     [NumRegs]int32
	Conf   [NumRegs]int32
	Stack  [4096]int32
	SP     int
	PC     int
	Halted bool
	Cycles int
	StripN int
}

func NewVM() *VM {
	return &VM{SP: 4096}
}

func formatSize(op uint8) int {
	switch {
	case op <= 0x07: return 1
	case op <= 0x0F: return 2
	case op <= 0x17: return 2
	case op <= 0x1F: return 3
	case op <= 0x4F: return 4
	default: return 1
	}
}

func (vm *VM) Execute(bc []byte) int {
	for !vm.Halted && vm.PC < len(bc) {
		result := vm.step(bc)
		if result < 0 {
			break
		}
		// result == 0: jump already set PC
		// result > 0: normal instruction, advance PC
		if result > 0 {
			vm.PC += result
		}
	}
	return vm.Cycles
}

func (vm *VM) step(bc []byte) int {
	if vm.PC >= len(bc) {
		return -1
	}
	op := bc[vm.PC]
	size := formatSize(op)
	if vm.PC+size > len(bc) {
		vm.Halted = true
		return -1
	}
	vm.Cycles++
	if vm.StripN > 0 {
		vm.StripN--
	}

	switch op {
	case OpHALT:
		vm.Halted = true
		return 1
	case OpNOP:
		return 1
	case OpINC:
		vm.GP[bc[vm.PC+1]]++
		return 2
	case OpDEC:
		vm.GP[bc[vm.PC+1]]--
		return 2
	case OpNOT:
		vm.GP[bc[vm.PC+1]] = ^vm.GP[bc[vm.PC+1]]
		return 2
	case OpNEG:
		vm.GP[bc[vm.PC+1]] = -vm.GP[bc[vm.PC+1]]
		return 2
	case OpPUSH:
		vm.SP--
		vm.Stack[vm.SP] = vm.GP[bc[vm.PC+1]]
		return 2
	case OpPOP:
		vm.GP[bc[vm.PC+1]] = vm.Stack[vm.SP]
		vm.SP++
		return 2
	case OpSTRIPCONF:
		vm.StripN = int(bc[vm.PC+1])
		return 2
	case OpMOVI:
		vm.GP[bc[vm.PC+1]] = int32(int8(bc[vm.PC+2]))
		return 3
	case OpADDI:
		vm.GP[bc[vm.PC+1]] += int32(int8(bc[vm.PC+2]))
		return 3
	case OpSUBI:
		vm.GP[bc[vm.PC+1]] -= int32(int8(bc[vm.PC+2]))
		return 3
	case OpADD:
		vm.GP[bc[vm.PC+1]] = vm.GP[bc[vm.PC+2]] + vm.GP[bc[vm.PC+3]]
		return 4
	case OpSUB:
		vm.GP[bc[vm.PC+1]] = vm.GP[bc[vm.PC+2]] - vm.GP[bc[vm.PC+3]]
		return 4
	case OpMUL:
		vm.GP[bc[vm.PC+1]] = vm.GP[bc[vm.PC+2]] * vm.GP[bc[vm.PC+3]]
		return 4
	case OpDIV:
		if vm.GP[bc[vm.PC+3]] == 0 {
			vm.Halted = true
			return 4
		}
		vm.GP[bc[vm.PC+1]] = vm.GP[bc[vm.PC+2]] / vm.GP[bc[vm.PC+3]]
		return 4
	case OpMOD:
		vm.GP[bc[vm.PC+1]] = vm.GP[bc[vm.PC+2]] % vm.GP[bc[vm.PC+3]]
		return 4
	case OpAND:
		vm.GP[bc[vm.PC+1]] = vm.GP[bc[vm.PC+2]] & vm.GP[bc[vm.PC+3]]
		return 4
	case OpOR:
		vm.GP[bc[vm.PC+1]] = vm.GP[bc[vm.PC+2]] | vm.GP[bc[vm.PC+3]]
		return 4
	case OpMOV:
		vm.GP[bc[vm.PC+1]] = vm.GP[bc[vm.PC+2]]
		return 4
	case OpJNZ:
		rd := bc[vm.PC+1]
		off := int32(int8(bc[vm.PC+2]))
		if vm.GP[rd] != 0 {
			vm.PC += int(off)
			return 0 // jump, don't advance
		}
		return 4
	case OpJZ:
		rd := bc[vm.PC+1]
		off := int32(int8(bc[vm.PC+2]))
		if vm.GP[rd] == 0 {
			vm.PC += int(off)
			return 0
		}
		return 4
	case OpMOVI16:
		rd := bc[vm.PC+1]
		imm := int16(uint16(bc[vm.PC+2]) | uint16(bc[vm.PC+3])<<8)
		vm.GP[rd] = int32(imm)
		return 4
	case OpJMP:
		off := int16(uint16(bc[vm.PC+2]) | uint16(bc[vm.PC+3])<<8)
		vm.PC += int(off)
		return 0
	case OpLOOP:
		rd := bc[vm.PC+1]
		off := int(uint16(bc[vm.PC+2]) | uint16(bc[vm.PC+3])<<8)
		vm.GP[rd]--
		if vm.GP[rd] > 0 {
			vm.PC -= off
			return 0
		}
		return 4
	default:
		return size
	}
}
