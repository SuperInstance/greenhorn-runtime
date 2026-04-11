package flux

import (
	"testing"
)

func TestHalt(t *testing.T) {
	vm := NewVM()
	bc := []byte{OpHALT}
	vm.Execute(bc)
	if !vm.Halted { t.Fatal("should halt") }
	if vm.Cycles != 1 { t.Fatalf("cycles: %d", vm.Cycles) }
}

func TestMovi(t *testing.T) {
	vm := NewVM()
	bc := []byte{OpMOVI, 0, 42, OpHALT}
	vm.Execute(bc)
	if vm.GP[0] != 42 { t.Fatalf("R0=%d", vm.GP[0]) }
}

func TestMoviNeg(t *testing.T) {
	vm := NewVM()
	bc := []byte{OpMOVI, 0, 0xFB, OpHALT} // -5
	vm.Execute(bc)
	if vm.GP[0] != -5 { t.Fatalf("R0=%d", vm.GP[0]) }
}

func TestMovi16(t *testing.T) {
	vm := NewVM()
	bc := []byte{OpMOVI16, 0, 0xE8, 0x03, OpHALT} // 1000
	vm.Execute(bc)
	if vm.GP[0] != 1000 { t.Fatalf("R0=%d", vm.GP[0]) }
}

func TestAdd(t *testing.T) {
	vm := NewVM()
	bc := []byte{
		OpMOVI, 0, 10,
		OpMOVI, 1, 20,
		OpADD, 2, 0, 1,
		OpHALT,
	}
	vm.Execute(bc)
	if vm.GP[2] != 30 { t.Fatalf("R2=%d", vm.GP[2]) }
}

func TestFibonacci(t *testing.T) {
	vm := NewVM()
	// R0=1, R1=1, R2=10
	// loop: R3=R0+R1, R0=R1, R1=R3, R2--, JNZ R2 loop
	bc := []byte{
		OpMOVI, 0, 1,       // 0: R0=1
		OpMOVI, 1, 1,       // 3: R1=1
		OpMOVI, 2, 10,      // 6: R2=10
		OpADD, 3, 0, 1,     // 9: R3=R0+R1
		OpMOV, 0, 1, 0,     // 13: R0=R1
		OpMOV, 1, 3, 0,     // 17: R1=R3
		OpDEC, 2,           // 21: R2--
		OpJNZ, 2, 0xF2, 0,  // 23: JNZ R2, -14 → back to ADD at 9
		OpHALT,
	}
	vm.Execute(bc)
	// 10 iterations: 1,1,2,3,5,8,13,21,34,55,89,144
	if vm.GP[1] != 144 { t.Fatalf("fib(10)=%d", vm.GP[1]) }
}

func TestIncDec(t *testing.T) {
	vm := NewVM()
	bc := []byte{
		OpMOVI, 0, 10,
		OpINC, 0,
		OpINC, 0,
		OpDEC, 0,
		OpHALT,
	}
	vm.Execute(bc)
	if vm.GP[0] != 11 { t.Fatalf("R0=%d", vm.GP[0]) }
}

func TestPushPop(t *testing.T) {
	vm := NewVM()
	bc := []byte{
		OpMOVI, 0, 42,
		OpPUSH, 0,
		OpMOVI, 0, 0,
		OpPOP, 1,
		OpHALT,
	}
	vm.Execute(bc)
	if vm.GP[1] != 42 { t.Fatalf("R1=%d", vm.GP[1]) }
}

func TestStripConf(t *testing.T) {
	vm := NewVM()
	bc := []byte{
		OpSTRIPCONF, 3,
		OpMOVI, 0, 1,
		OpHALT,
	}
	vm.Execute(bc)
	if vm.StripN != 1 { t.Fatalf("strip=%d", vm.StripN) }
}
