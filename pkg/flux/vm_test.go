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

// --- New tests for untested opcodes and edge cases ---

func TestNewVM(t *testing.T) {
        vm := NewVM()
        if vm.SP != 4096 {
                t.Fatalf("SP should start at 4096, got %d", vm.SP)
        }
        if vm.Halted {
                t.Fatal("should not start halted")
        }
        if vm.PC != 0 {
                t.Fatalf("PC should start at 0, got %d", vm.PC)
        }
        if vm.Cycles != 0 {
                t.Fatalf("Cycles should start at 0, got %d", vm.Cycles)
        }
}

func TestNOP(t *testing.T) {
        vm := NewVM()
        bc := []byte{OpNOP, OpNOP, OpHALT}
        vm.Execute(bc)
        if vm.Cycles != 3 {
                t.Fatalf("expected 3 cycles, got %d", vm.Cycles)
        }
}

func TestSub(t *testing.T) {
        vm := NewVM()
        bc := []byte{
                OpMOVI, 0, 30,
                OpMOVI, 1, 12,
                OpSUB, 2, 0, 1,
                OpHALT,
        }
        vm.Execute(bc)
        if vm.GP[2] != 18 {
                t.Fatalf("R2=%d, want 18", vm.GP[2])
        }
}

func TestMul(t *testing.T) {
        vm := NewVM()
        bc := []byte{
                OpMOVI, 0, 7,
                OpMOVI, 1, 6,
                OpMUL, 2, 0, 1,
                OpHALT,
        }
        vm.Execute(bc)
        if vm.GP[2] != 42 {
                t.Fatalf("R2=%d, want 42", vm.GP[2])
        }
}

func TestDiv(t *testing.T) {
        vm := NewVM()
        bc := []byte{
                OpMOVI, 0, 100,
                OpMOVI, 1, 7,
                OpDIV, 2, 0, 1,
                OpHALT,
        }
        vm.Execute(bc)
        if vm.GP[2] != 14 {
                t.Fatalf("R2=%d, want 14 (100/7 truncated)", vm.GP[2])
        }
}

func TestDivByZero(t *testing.T) {
        vm := NewVM()
        bc := []byte{
                OpMOVI, 0, 10,
                OpMOVI, 1, 0,
                OpDIV, 2, 0, 1,
                OpHALT,
        }
        vm.Execute(bc)
        if !vm.Halted {
                t.Fatal("should halt on division by zero")
        }
}

func TestMod(t *testing.T) {
        vm := NewVM()
        bc := []byte{
                OpMOVI, 0, 17,
                OpMOVI, 1, 5,
                OpMOD, 2, 0, 1,
                OpHALT,
        }
        vm.Execute(bc)
        if vm.GP[2] != 2 {
                t.Fatalf("R2=%d, want 2 (17%%5)", vm.GP[2])
        }
}

func TestAndOr(t *testing.T) {
        vm := NewVM()
        bc := []byte{
                OpMOVI16, 0, 0xFF, 0x00, // R0 = 255
                OpMOVI16, 1, 0x0F, 0x00, // R1 = 15
                OpAND, 2, 0, 1,           // R2 = 255 & 15 = 15
                OpMOVI16, 0, 0xF0, 0x00, // R0 = 240
                OpOR, 3, 0, 1,            // R3 = 240 | 15 = 255
                OpHALT,
        }
        vm.Execute(bc)
        if vm.GP[2] != 15 {
                t.Fatalf("AND: R2=%d, want 15", vm.GP[2])
        }
        if vm.GP[3] != 255 {
                t.Fatalf("OR: R3=%d, want 255", vm.GP[3])
        }
}

func TestNot(t *testing.T) {
        vm := NewVM()
        bc := []byte{
                OpMOVI, 0, 0,      // R0 = 0
                OpNOT, 0,           // R0 = ~0 = -1
                OpHALT,
        }
        vm.Execute(bc)
        if vm.GP[0] != -1 {
                t.Fatalf("NOT: R0=%d, want -1", vm.GP[0])
        }
}

func TestNeg(t *testing.T) {
        vm := NewVM()
        bc := []byte{
                OpMOVI, 0, 42,
                OpNEG, 0,
                OpHALT,
        }
        vm.Execute(bc)
        if vm.GP[0] != -42 {
                t.Fatalf("NEG: R0=%d, want -42", vm.GP[0])
        }
}

func TestAddiSubi(t *testing.T) {
        vm := NewVM()
        bc := []byte{
                OpMOVI, 0, 100,
                OpADDI, 0, 10,    // R0 = 110
                OpSUBI, 0, 25,    // R0 = 85 (0xE7 = -25 as int8)
                OpHALT,
        }
        vm.Execute(bc)
        if vm.GP[0] != 85 {
                t.Fatalf("R0=%d, want 85", vm.GP[0])
        }
}

func TestSubiNegative(t *testing.T) {
        vm := NewVM()
        bc := []byte{
                OpMOVI, 0, 10,
                OpSUBI, 0, 0xFB,  // -5 → R0 = 10 + (-(-5)) = 10 - (-5)... wait, SUBI subtracts int8
                // 0xFB as int8 = -5, so R0 = R0 - (-5) = R0 + 5 = 15
                OpHALT,
        }
        vm.Execute(bc)
        if vm.GP[0] != 15 {
                t.Fatalf("R0=%d, want 15", vm.GP[0])
        }
}

func TestMOV(t *testing.T) {
        vm := NewVM()
        bc := []byte{
                OpMOVI16, 0, 0x88, 0x13, // R0 = 5000
                OpMOV, 5, 0, 0,          // R5 = R0
                OpHALT,
        }
        vm.Execute(bc)
        if vm.GP[5] != 5000 {
                t.Fatalf("MOV: R5=%d, want 5000", vm.GP[5])
        }
}

func TestJZ(t *testing.T) {
        vm := NewVM()
        // JZ with zero value should jump
        bc := []byte{
                OpMOVI, 0, 0,        // 0-2: R0=0
                OpMOVI, 1, 99,       // 3-5: R1=99
                OpJZ, 0, 7, 0,       // 6-9: JZ R0, offset=7 → PC=13 (skip MOVI below)
                OpMOVI, 1, 0,        // 10-12: should be skipped
                OpHALT,               // 13
        }
        vm.Execute(bc)
        if vm.GP[1] != 99 {
                t.Fatalf("JZ should have skipped: R1=%d, want 99", vm.GP[1])
        }
}

func TestJZNotTaken(t *testing.T) {
        vm := NewVM()
        // JZ with non-zero value should NOT jump
        bc := []byte{
                OpMOVI, 0, 1,
                OpMOVI, 1, 0,
                OpJZ, 0, 4, 0,   // should NOT jump
                OpMOVI, 1, 42,    // should execute
                OpHALT,
        }
        vm.Execute(bc)
        if vm.GP[1] != 42 {
                t.Fatalf("JZ not taken: R1=%d, want 42", vm.GP[1])
        }
}

func TestJMP(t *testing.T) {
        vm := NewVM()
        // JMP forward to skip instruction
        bc := []byte{
                OpMOVI, 0, 0,
                OpJMP, 0, 4, 0,  // skip 1 instruction (forward)
                OpMOVI, 0, 99,   // should be skipped
                OpMOVI, 0, 42,   // should execute
                OpHALT,
        }
        vm.Execute(bc)
        if vm.GP[0] != 42 {
                t.Fatalf("JMP: R0=%d, want 42", vm.GP[0])
        }
}

func TestLOOP(t *testing.T) {
        vm := NewVM()
        // LOOP decrements R1, jumps back if R1 > 0
        // R1=5: decrements to 4,3,2,1,0 → 5 INC iterations → R0=5
        bc := []byte{
                OpMOVI, 0, 0,      // 0-2: R0 = 0
                OpMOVI, 1, 5,      // 3-5: R1 = 5
                OpINC, 0,           // 6-7: R0++
                OpLOOP, 1, 2, 0,   // 8-11: R1--, if R1 > 0 jump back 2 to INC at 6
                OpHALT,             // 12
        }
        vm.Execute(bc)
        if vm.GP[0] != 5 {
                t.Fatalf("LOOP sum: R0=%d, want 5", vm.GP[0])
        }
        if vm.GP[1] != 0 {
                t.Fatalf("LOOP counter: R1=%d, want 0", vm.GP[1])
        }
}

func TestEmptyBytecode(t *testing.T) {
        vm := NewVM()
        cycles := vm.Execute([]byte{})
        if cycles != 0 {
                t.Fatalf("expected 0 cycles for empty bytecode, got %d", cycles)
        }
}

func TestTruncatedBytecode(t *testing.T) {
        vm := NewVM()
        // OpADD needs 4 bytes but we only provide 2
        bc := []byte{OpADD, 0}
        vm.Execute(bc)
        if !vm.Halted {
                t.Fatal("should halt on truncated instruction")
        }
}

func TestMovi16Negative(t *testing.T) {
        vm := NewVM()
        // -1000 as int16 = 0xFC18
        bc := []byte{OpMOVI16, 1, 0x18, 0xFC, OpHALT}
        vm.Execute(bc)
        if vm.GP[1] != -1000 {
                t.Fatalf("R1=%d, want -1000", vm.GP[1])
        }
}

func TestMultiplePushPop(t *testing.T) {
        vm := NewVM()
        bc := []byte{
                OpMOVI, 0, 10,
                OpMOVI, 1, 20,
                OpMOVI, 2, 30,
                OpPUSH, 0,
                OpPUSH, 1,
                OpPUSH, 2,
                OpPOP, 3,
                OpPOP, 4,
                OpPOP, 5,
                OpHALT,
        }
        vm.Execute(bc)
        if vm.GP[3] != 30 { t.Fatalf("R3=%d, want 30", vm.GP[3]) }
        if vm.GP[4] != 20 { t.Fatalf("R4=%d, want 20", vm.GP[4]) }
        if vm.GP[5] != 10 { t.Fatalf("R5=%d, want 10", vm.GP[5]) }
}

func TestAddiNegative(t *testing.T) {
        vm := NewVM()
        // 0xFB as int8 = -5
        bc := []byte{
                OpMOVI, 0, 10,
                OpADDI, 0, 0xFB,  // R0 = 10 + (-5) = 5
                OpHALT,
        }
        vm.Execute(bc)
        if vm.GP[0] != 5 {
                t.Fatalf("R0=%d, want 5", vm.GP[0])
        }
}
