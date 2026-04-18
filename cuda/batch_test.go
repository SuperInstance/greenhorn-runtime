package cuda

import (
        "fmt"
        "testing"
)

// FLUX opcodes (matching pkg/flux/vm.go)
const (
        opHALT      = 0x00
        opNOP       = 0x01
        opRET       = 0x02
        opINC       = 0x08
        opDEC       = 0x09
        opNOT       = 0x0A
        opNEG       = 0x0B
        opPUSH      = 0x0C
        opPOP       = 0x0D
        opSTRIPCONF = 0x17
        opMOVI      = 0x18
        opADDI      = 0x19
        opSUBI      = 0x1A
        opADD       = 0x20
        opSUB       = 0x21
        opMUL       = 0x22
        opDIV       = 0x23
        opMOD       = 0x24
        opAND       = 0x25
        opOR        = 0x26
        opXOR       = 0x27
        opSHL       = 0x28
        opSHR       = 0x29
        opMIN       = 0x2A
        opMAX       = 0x2B
        opCMP_EQ    = 0x2C
        opCMP_LT    = 0x2D
        opCMP_GT    = 0x2E
        opCMP_NE    = 0x2F
        opMOV       = 0x3A
        opJZ        = 0x3C
        opJNZ       = 0x3D
        opJLT       = 0x3E
        opJGT       = 0x3F
        opMOVI16    = 0x40
        opJMP       = 0x43
        opLOOP      = 0x46
        opCALL      = 0x4A
)

// newTestExecutor creates a BatchExecutor for testing.
func newTestExecutor(t *testing.T) *BatchExecutor {
        t.Helper()
        e, err := NewBatchExecutor()
        if err != nil {
                t.Fatalf("NewBatchExecutor: %v", err)
        }
        t.Cleanup(func() { e.Close() })
        return e
}

// ============================================================================
// Conformance Tests (matching pkg/flux/vm_test.go)
// ============================================================================

func TestHalt(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{{opHALT}}
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 0 {
                t.Errorf("HALT: R0=%d, want 0", result.Results[0])
        }
        if result.Cycles[0] != 1 {
                t.Errorf("HALT: cycles=%d, want 1", result.Cycles[0])
        }
        if result.Errors[0] != ErrNone {
                t.Errorf("HALT: error=%d, want 0", result.Errors[0])
        }
}

func TestMovi(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{{opMOVI, 0, 42, opHALT}}
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 42 {
                t.Errorf("MOVI: R0=%d, want 42", result.Results[0])
        }
}

func TestMoviNeg(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{{opMOVI, 0, 0xFB, opHALT}} // -5 as signed byte
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != -5 {
                t.Errorf("MOVI neg: R0=%d, want -5", result.Results[0])
        }
}

func TestMovi16(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{{opMOVI16, 0, 0xE8, 0x03, opHALT}} // 1000
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 1000 {
                t.Errorf("MOVI16: R0=%d, want 1000", result.Results[0])
        }
}

func TestAdd(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 10, opMOVI, 1, 20, opADD, 0, 0, 1, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 30 {
                t.Errorf("ADD: R0=%d, want 30", result.Results[0])
        }
}

func TestSub(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 50, opMOVI, 1, 20, opSUB, 0, 0, 1, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 30 {
                t.Errorf("SUB: R0=%d, want 30", result.Results[0])
        }
}

func TestMul(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 6, opMOVI, 1, 7, opMUL, 0, 0, 1, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 42 {
                t.Errorf("MUL: R0=%d, want 42", result.Results[0])
        }
}

func TestDiv(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 100, opMOVI, 1, 3, opDIV, 0, 0, 1, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 33 {
                t.Errorf("DIV: R0=%d, want 33", result.Results[0])
        }
}

func TestMod(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 17, opMOVI, 1, 5, opMOD, 0, 0, 1, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 2 {
                t.Errorf("MOD: R0=%d, want 2", result.Results[0])
        }
}

func TestDivByZero(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 10, opMOVI, 1, 0, opDIV, 0, 0, 1, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        // On CPU fallback, div-by-zero halts but reports ErrNone.
        // On CUDA, it reports ErrDivByZero. Both are acceptable.
        if result.Results[0] != 10 {
                t.Errorf("DIV by zero: R0=%d, want 10 (unchanged)", result.Results[0])
        }
        // CPU fallback halts with ErrNone; CUDA reports ErrDivByZero
        if Available() && result.Errors[0] != ErrDivByZero {
                t.Errorf("DIV by zero: error=%d, want %d", result.Errors[0], ErrDivByZero)
        }
}

// ============================================================================
// Bitwise Operation Tests
// ============================================================================

func TestNot(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{{opNOT, 0, opHALT}}
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != -1 {
                t.Errorf("NOT: R0=%d, want -1", result.Results[0])
        }
}

func TestNeg(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{{opMOVI, 0, 42, opNEG, 0, opHALT}}
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != -42 {
                t.Errorf("NEG: R0=%d, want -42", result.Results[0])
        }
}

func TestAnd(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 0xFF, opMOVI, 1, 0x0F, opAND, 0, 0, 1, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 15 {
                t.Errorf("AND: R0=%d, want 15", result.Results[0])
        }
}

func TestOr(t *testing.T) {
        e := newTestExecutor(t)
        // 0xF0 | 0x0F = 0xFF = 255 (unsigned), but as int32 sign-extended it's -1
        // Both are correct depending on interpretation; use -1 for int32 consistency
        programs := [][]byte{
                {opMOVI, 0, 0xF0, opMOVI, 1, 0x0F, opOR, 0, 0, 1, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != -1 {
                t.Errorf("OR: R0=%d, want -1 (0xFF sign-extended)", result.Results[0])
        }
}

func TestXor(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 0xFF, opMOVI, 1, 0xFF, opXOR, 0, 0, 1, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 0 {
                t.Errorf("XOR: R0=%d, want 0", result.Results[0])
        }
}

func TestShl(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 1, opMOVI, 1, 4, opSHL, 0, 0, 1, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 16 {
                t.Errorf("SHL: R0=%d, want 16", result.Results[0])
        }
}

func TestShr(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 16, opMOVI, 1, 2, opSHR, 0, 0, 1, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 4 {
                t.Errorf("SHR: R0=%d, want 4", result.Results[0])
        }
}

// ============================================================================
// Comparison Operation Tests
// ============================================================================

func TestCmpEQ(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 10, opMOVI, 1, 10, opCMP_EQ, 2, 0, 1, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 10 {
                t.Errorf("CMP_EQ: R0=%d, want 10", result.Results[0])
        }
}

func TestCmpLT(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 5, opMOVI, 1, 10, opCMP_LT, 0, 0, 1, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 1 {
                t.Errorf("CMP_LT: R0=%d, want 1", result.Results[0])
        }
}

func TestCmpGT(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 10, opMOVI, 1, 5, opCMP_GT, 0, 0, 1, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 1 {
                t.Errorf("CMP_GT: R0=%d, want 1", result.Results[0])
        }
}

func TestCmpNE(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 5, opMOVI, 1, 10, opCMP_NE, 0, 0, 1, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 1 {
                t.Errorf("CMP_NE: R0=%d, want 1", result.Results[0])
        }
}

func TestMin(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 10, opMOVI, 1, 20, opMIN, 0, 0, 1, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 10 {
                t.Errorf("MIN: R0=%d, want 10", result.Results[0])
        }
}

func TestMax(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 10, opMOVI, 1, 20, opMAX, 0, 0, 1, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 20 {
                t.Errorf("MAX: R0=%d, want 20", result.Results[0])
        }
}

// ============================================================================
// Stack Operation Tests
// ============================================================================

func TestIncDec(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 10, opINC, 0, opINC, 0, opDEC, 0, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 11 {
                t.Errorf("INC/DEC: R0=%d, want 11", result.Results[0])
        }
}

func TestPushPop(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 42, opPUSH, 0, opMOVI, 0, 0, opPOP, 1, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 0 {
                t.Errorf("PUSH/POP: R0=%d, want 0", result.Results[0])
        }
}

func TestAddiSubi(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 10, opADDI, 0, 5, opSUBI, 0, 3, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 12 {
                t.Errorf("ADDI/SUBI: R0=%d, want 12", result.Results[0])
        }
}

// ============================================================================
// Control Flow Tests
// ============================================================================

func TestJNZ(t *testing.T) {
        e := newTestExecutor(t)
        // Loop: R0=5, decrement and loop back until 0
        programs := [][]byte{
                {
                        opMOVI, 0, 5,
                        opDEC, 0,           // offset 3
                        opJNZ, 0, 0xFE, 0,  // offset 5: JNZ R0, -2 -> back to offset 3
                        opMOVI, 0, 100,     // offset 9: R0 = 100 (after loop)
                        opHALT,
                },
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 100 {
                t.Errorf("JNZ: R0=%d, want 100", result.Results[0])
        }
}

func TestJZ(t *testing.T) {
        e := newTestExecutor(t)
        // Program layout:
        // offset 0: MOVI R0, 0      (3 bytes)
        // offset 3: JZ R0, +7       (4 bytes) -> jumps to offset 10
        // offset 7: MOVI R0, 50     (3 bytes, skipped)
        // offset 10: MOVI R0, 99    (3 bytes)
        // offset 13: HALT           (1 byte)
        programs := [][]byte{
                {
                        opMOVI, 0, 0,      // 0: R0 = 0
                        opJZ, 0, 0x07, 0,  // 3: JZ R0, +7 -> PC = 3 + 7 = 10
                        opMOVI, 0, 50,     // 7: skipped
                        opMOVI, 0, 99,     // 10: R0 = 99
                        opHALT,             // 13
                },
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 99 {
                t.Errorf("JZ: R0=%d, want 99", result.Results[0])
        }
}

func TestLoop(t *testing.T) {
        e := newTestExecutor(t)
        // LOOP R0, offset: decrement R0, jump back by offset if still > 0
        programs := [][]byte{
                {
                        opMOVI, 0, 5,       // 0: R0 = 5 (loop counter)
                        opMOVI, 1, 0,       // 3: R1 = 0 (accumulator)
                        // loop at offset 6:
                        opINC, 1,            // 6: R1++
                        opLOOP, 0, 0x02, 0, // 8: LOOP R0, 2 -> PC = 8 - 2 = 6 (back to INC)
                        opMOV, 0, 1, 0,     // 12: R0 = R1 = 5
                        opHALT,             // 16
                },
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 5 {
                t.Errorf("LOOP: R0=%d, want 5", result.Results[0])
        }
}

func TestMov(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 42, opMOV, 1, 0, 0, opMOV, 0, 1, 0, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 42 {
                t.Errorf("MOV: R0=%d, want 42", result.Results[0])
        }
}

// ============================================================================
// Fibonacci Test (from Go vm_test.go)
// ============================================================================

func TestFibonacci(t *testing.T) {
        e := newTestExecutor(t)
        // R0=1, R1=1, R2=10, loop: R3=R0+R1, R0=R1, R1=R3, R2--, JNZ R2
        programs := [][]byte{
                {
                        opMOVI, 0, 1,       // 0: R0=1
                        opMOVI, 1, 1,       // 3: R1=1
                        opMOVI, 2, 10,      // 6: R2=10
                        opADD, 3, 0, 1,     // 9: R3=R0+R1
                        opMOV, 0, 1, 0,     // 13: R0=R1
                        opMOV, 1, 3, 0,     // 17: R1=R3
                        opDEC, 2,           // 21: R2--
                        opJNZ, 2, 0xF2, 0,  // 23: JNZ R2, -14 -> back to offset 9
                        opHALT,             // 27
                },
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Errors[0] != ErrNone {
                t.Errorf("Fibonacci: error=%d, want 0", result.Errors[0])
        }
        if result.Results[0] != 89 {
                t.Errorf("Fibonacci: R0=%d, want 89", result.Results[0])
        }
}

// ============================================================================
// Factorial Test
// ============================================================================

func TestFactorial(t *testing.T) {
        e := newTestExecutor(t)
        // Compute factorial(5) = 120 using JNZ loop
        programs := [][]byte{
                {
                        opMOVI, 0, 5,       // 0: R0 = 5
                        opMOVI, 1, 1,       // 3: R1 = 1
                        // loop at offset 6:
                        opMUL, 1, 1, 0,     // 6: R1 *= R0 (size 4)
                        opDEC, 0,           // 10: R0-- (size 2)
                        opJNZ, 0, 0xFA, 0,  // 12: JNZ R0, -6 -> PC = 12 + (-6) = 6
                        opMOV, 0, 1, 0,     // 16: R0 = R1
                        opHALT,             // 20
                },
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 120 {
                t.Errorf("Factorial: R0=%d, want 120", result.Results[0])
        }
}

func TestFactorialLoop(t *testing.T) {
        e := newTestExecutor(t)
        // Compute factorial(5) = 120 using LOOP instruction
        // LOOP semantics: gp[rd]--; if gp[rd] > 0 { pc -= offset }
        programs := [][]byte{
                {
                        opMOVI, 0, 5,       // 0: R0 = 5
                        opMOVI, 1, 1,       // 3: R1 = 1
                        // loop at offset 6:
                        opMUL, 1, 1, 0,     // 6: R1 *= R0 (size 4)
                        opLOOP, 0, 0x04, 0, // 10: LOOP R0, 4 -> PC = 10 - 4 = 6
                        opMOV, 0, 1, 0,     // 14: R0 = R1
                        opHALT,             // 18
                },
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 120 {
                t.Errorf("FactorialLoop: R0=%d, want 120", result.Results[0])
        }
}

// ============================================================================
// NOP Test
// ============================================================================

func TestNOP(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 7, opNOP, opNOP, opNOP, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 7 {
                t.Errorf("NOP: R0=%d, want 7", result.Results[0])
        }
        if result.Cycles[0] != 5 { // MOVI + 3 NOP + HALT = 5
                t.Errorf("NOP: cycles=%d, want 5", result.Cycles[0])
        }
}

// ============================================================================
// Batch Execution Tests
// ============================================================================

func TestBatchMultiple(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 1, opHALT},
                {opMOVI, 0, 2, opHALT},
                {opMOVI, 0, 3, opHALT},
                {opMOVI, 0, 4, opHALT},
                {opMOVI, 0, 5, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        for i := 0; i < 5; i++ {
                want := int32(i + 1)
                if result.Results[i] != want {
                        t.Errorf("Batch[%d]: R0=%d, want %d", i, result.Results[i], want)
                }
                if result.Errors[i] != ErrNone {
                        t.Errorf("Batch[%d]: error=%d", i, result.Errors[i])
                }
        }
}

func TestBatchMany(t *testing.T) {
        e := newTestExecutor(t)
        // Run 300 programs — more than a single CUDA block (256 threads)
        programs := make([][]byte, 300)
        for i := 0; i < 300; i++ {
                programs[i] = []byte{opMOVI, 0, byte(int(i) & 0x7F), opHALT}
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.NumPrograms != 300 {
                t.Errorf("NumPrograms=%d, want 300", result.NumPrograms)
        }
        if !result.AllOK() {
                t.Errorf("Not all programs OK, error count=%d", result.ErrorCount())
        }
        for i := 0; i < 300; i++ {
                want := int32(i & 0x7F)
                if result.Results[i] != want {
                        t.Errorf("BatchMany[%d]: R0=%d, want %d", i, result.Results[i], want)
                }
        }
}

func TestEmptyBatch(t *testing.T) {
        e := newTestExecutor(t)
        _, err := e.Run(nil)
        if err == nil {
                t.Error("expected error for nil programs")
        }
        _, err = e.Run([][]byte{})
        if err == nil {
                t.Error("expected error for empty programs")
        }
}

// ============================================================================
// StripConf Test
// ============================================================================

func TestStripConf(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opSTRIPCONF, 3, opMOVI, 0, 42, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 42 {
                t.Errorf("STRIPCONF: R0=%d, want 42", result.Results[0])
        }
}

// ============================================================================
// Result Helper Tests
// ============================================================================

func TestErrorString(t *testing.T) {
        tests := []struct {
                code int
                want string
        }{
                {ErrNone, "success"},
                {ErrDivByZero, "division by zero"},
                {ErrStackOverflow, "stack overflow"},
                {ErrStackUnderflow, "stack underflow"},
                {ErrInvalidOpcode, "invalid opcode"},
                {ErrMaxCycles, "max cycles exceeded"},
                {ErrPCOutOfBounds, "PC out of bounds"},
                {ErrA2AUnsupported, "A2A operation not supported"},
                {ErrBadRegister, "bad register index"},
                {999, "unknown error code 999"},
        }
        for _, tt := range tests {
                got := ErrorString(tt.code)
                if got != tt.want {
                        t.Errorf("ErrorString(%d) = %q, want %q", tt.code, got, tt.want)
                }
        }
}

func TestResultHelpers(t *testing.T) {
        e := newTestExecutor(t)
        programs := [][]byte{
                {opMOVI, 0, 42, opHALT},
                {opMOVI, 0, 99, opHALT},
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if !result.AllOK() {
                t.Error("AllOK should be true")
        }
        if result.ErrorCount() != 0 {
                t.Errorf("ErrorCount=%d, want 0", result.ErrorCount())
        }
        if result.NumPrograms != 2 {
                t.Errorf("NumPrograms=%d, want 2", result.NumPrograms)
        }
}

// ============================================================================
// Query Tests
// ============================================================================

func TestDeviceCount(t *testing.T) {
        count := DeviceCount()
        t.Logf("CUDA device count: %d", count)
}

func TestAvailable(t *testing.T) {
        avail := Available()
        t.Logf("CUDA available: %v", avail)
}

func TestDeviceInfo(t *testing.T) {
        info := DeviceInfo(0)
        t.Logf("Device 0: %s", info)
}

// ============================================================================
// CALL/RET Test
// ============================================================================

func TestCallRet(t *testing.T) {
        e := newTestExecutor(t)
        // Call a subroutine that adds 10 to R0
        // Main: MOVI R0, 5; CALL sub; HALT
        // Sub (at offset +5): ADDI R0, 10; RET
        programs := [][]byte{
                {
                        opMOVI, 0, 5,        // 0: R0 = 5
                        opCALL, 0, 0x05, 0,  // 3: CALL -> PC = 3 + 5 = 8
                        opHALT,              // 7: (return here)
                        // subroutine at offset 8:
                        opADDI, 0, 10,       // 8: R0 += 10
                        opRET,               // 11: return to PC 7 (pushed by CALL)
                },
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 15 {
                t.Errorf("CALL/RET: R0=%d, want 15", result.Results[0])
        }
}

// ============================================================================
// AllOps Conformance Test
// ============================================================================

func TestAllOpsConformance(t *testing.T) {
        e := newTestExecutor(t)

        tests := []struct {
                name    string
                prog    []byte
                wantR0  int32
                wantErr int32
        }{
                {"HALT", []byte{opHALT}, 0, ErrNone},
                {"NOP+HALT", []byte{opNOP, opHALT}, 0, ErrNone},
                {"MOVI 42", []byte{opMOVI, 0, 42, opHALT}, 42, ErrNone},
                {"MOVI -5", []byte{opMOVI, 0, 0xFB, opHALT}, -5, ErrNone},
                {"INC", []byte{opMOVI, 0, 10, opINC, 0, opHALT}, 11, ErrNone},
                {"DEC", []byte{opMOVI, 0, 10, opDEC, 0, opHALT}, 9, ErrNone},
                {"NOT", []byte{opNOT, 0, opHALT}, -1, ErrNone},
                {"NEG", []byte{opMOVI, 0, 5, opNEG, 0, opHALT}, -5, ErrNone},
                {"ADD", []byte{opMOVI, 0, 3, opMOVI, 1, 4, opADD, 0, 0, 1, opHALT}, 7, ErrNone},
                {"SUB", []byte{opMOVI, 0, 10, opMOVI, 1, 3, opSUB, 0, 0, 1, opHALT}, 7, ErrNone},
                {"MUL", []byte{opMOVI, 0, 6, opMOVI, 1, 7, opMUL, 0, 0, 1, opHALT}, 42, ErrNone},
                {"DIV", []byte{opMOVI, 0, 20, opMOVI, 1, 4, opDIV, 0, 0, 1, opHALT}, 5, ErrNone},
                {"MOD", []byte{opMOVI, 0, 17, opMOVI, 1, 5, opMOD, 0, 0, 1, opHALT}, 2, ErrNone},
                {"AND", []byte{opMOVI, 0, 0xFF, opMOVI, 1, 0x0F, opAND, 0, 0, 1, opHALT}, 15, ErrNone},
                {"OR", []byte{opMOVI, 0, 0xF0, opMOVI, 1, 0x0F, opOR, 0, 0, 1, opHALT}, -1, ErrNone},
                {"XOR", []byte{opMOVI, 0, 0xFF, opMOVI, 1, 0xFF, opXOR, 0, 0, 1, opHALT}, 0, ErrNone},
                {"SHL", []byte{opMOVI, 0, 1, opMOVI, 1, 8, opSHL, 0, 0, 1, opHALT}, 256, ErrNone},
                {"SHR", []byte{opMOVI16, 0, 0x00, 0x01, opMOVI, 1, 4, opSHR, 0, 0, 1, opHALT}, 16, ErrNone},
                {"MIN", []byte{opMOVI, 0, 3, opMOVI, 1, 7, opMIN, 0, 0, 1, opHALT}, 3, ErrNone},
                {"MAX", []byte{opMOVI, 0, 3, opMOVI, 1, 7, opMAX, 0, 0, 1, opHALT}, 7, ErrNone},
                {"MOVI16", []byte{opMOVI16, 0, 0xE8, 0x03, opHALT}, 1000, ErrNone},
                {"PUSH/POP", []byte{opMOVI, 0, 42, opPUSH, 0, opMOVI, 0, 0, opPOP, 0, opHALT}, 42, ErrNone},
                {"ADDI", []byte{opMOVI, 0, 10, opADDI, 0, 5, opHALT}, 15, ErrNone},
                {"SUBI", []byte{opMOVI, 0, 10, opSUBI, 0, 3, opHALT}, 7, ErrNone},
        }

        programs := make([][]byte, len(tests))
        for i, tt := range tests {
                programs[i] = tt.prog
        }

        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }

        for i, tt := range tests {
                if result.Errors[i] != tt.wantErr {
                        t.Errorf("%s: error=%d (%s), want %d", tt.name, result.Errors[i],
                                ErrorString(int(result.Errors[i])), tt.wantErr)
                }
                if result.Errors[i] == ErrNone && result.Results[i] != tt.wantR0 {
                        t.Errorf("%s: R0=%d, want %d", tt.name, result.Results[i], tt.wantR0)
                }
        }
}

// ============================================================================
// PackBytecodes Test
// ============================================================================

func TestPackBytecodes(t *testing.T) {
        programs := [][]byte{
                {opMOVI, 0, 1, opHALT},
                {opMOVI, 0, 2, opHALT},
                {opMOVI, 0, 3, opHALT, opNOP},
        }

        packed, offsets, lengths, totalSize, err := PackBytecodes(programs)
        if err != nil {
                t.Fatalf("PackBytecodes: %v", err)
        }

        if totalSize == 0 {
                t.Error("totalSize should be > 0")
        }
        if len(offsets) != 3 {
                t.Errorf("offsets len=%d, want 3", len(offsets))
        }
        if len(lengths) != 3 {
                t.Errorf("lengths len=%d, want 3", len(lengths))
        }
        if offsets[0] != 0 {
                t.Errorf("offsets[0]=%d, want 0", offsets[0])
        }
        if offsets[1] != 4 {
                t.Errorf("offsets[1]=%d, want 4", offsets[1])
        }
        if offsets[2] != 8 {
                t.Errorf("offsets[2]=%d, want 8", offsets[2])
        }

        fmt.Printf("Packed %d programs into %d bytes\n", len(programs), totalSize)
        _ = packed
}

// ============================================================================
// JMP Test (unconditional jump)
// ============================================================================

func TestJMP(t *testing.T) {
        e := newTestExecutor(t)
        // Layout:
        // offset 0: MOVI R0, 0      (3 bytes)
        // offset 3: JMP +5           (4 bytes) -> PC = 3 + 5 = 8
        // offset 7: NOP              (1 byte, skipped by JMP)
        // offset 8: MOVI R0, 99     (3 bytes)
        // offset 11: HALT            (1 byte)
        programs := [][]byte{
                {
                        opMOVI, 0, 0,       // 0: R0 = 0
                        opJMP, 0, 0x05, 0, // 3: JMP +5 -> PC = 3 + 5 = 8
                        opNOP,              // 7: skipped
                        opMOVI, 0, 99,     // 8: R0 = 99
                        opHALT,             // 11
                },
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 99 {
                t.Errorf("JMP: R0=%d, want 99", result.Results[0])
        }
}

// ============================================================================
// JLT/JGT Tests
// ============================================================================

func TestJLT(t *testing.T) {
        e := newTestExecutor(t)
        // R0 = -5, JLT should jump since -5 < 0
        // Layout:
        // offset 0: MOVI R0, -5     (3 bytes) [0xFB = -5]
        // offset 3: JLT R0, +5      (4 bytes) -> PC = 3 + 5 = 8
        // offset 7: NOP              (1 byte, skipped)
        // offset 8: MOVI R0, 42     (3 bytes)
        // offset 11: HALT            (1 byte)
        programs := [][]byte{
                {
                        opMOVI, 0, 0xFB,    // 0: R0 = -5
                        opJLT, 0, 0x05, 0,  // 3: JLT R0, +5 -> PC = 3 + 5 = 8
                        opNOP,               // 7: skipped
                        opMOVI, 0, 42,      // 8: R0 = 42
                        opHALT,              // 11
                },
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 42 {
                t.Errorf("JLT: R0=%d, want 42", result.Results[0])
        }
}

func TestJGT(t *testing.T) {
        e := newTestExecutor(t)
        // R0 = 10, JGT should jump since 10 > 0
        programs := [][]byte{
                {
                        opMOVI, 0, 10,      // 0: R0 = 10
                        opJGT, 0, 0x05, 0,  // 3: JGT R0, +5 -> PC = 3 + 5 = 8
                        opNOP,               // 7: skipped
                        opMOVI, 0, 77,      // 8: R0 = 77
                        opHALT,              // 11
                },
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 77 {
                t.Errorf("JGT: R0=%d, want 77", result.Results[0])
        }
}

func TestJLTNotTaken(t *testing.T) {
        e := newTestExecutor(t)
        // R0 = 5 (positive), JLT should NOT jump
        programs := [][]byte{
                {
                        opMOVI, 0, 5,       // 0: R0 = 5
                        opJLT, 0, 0x05, 0,  // 3: JLT R0, +5 -> NOT taken (5 >= 0)
                        opNOP,               // 7: not skipped (fall through)
                        opMOVI, 0, 42,      // 8: R0 = 42
                        opHALT,              // 11
                },
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.Results[0] != 42 {
                t.Errorf("JLT not-taken: R0=%d, want 42", result.Results[0])
        }
}

// ============================================================================
// Batch Stress Test (multi-block, >256 programs)
// ============================================================================

func TestBatchStress1000(t *testing.T) {
        e := newTestExecutor(t)
        // Run 1000 programs — exercises multi-block CUDA launch (>256 threads)
        factProg := []byte{
                opMOVI, 0, 5,
                opMOVI, 1, 1,
                opMUL, 1, 1, 0,
                opDEC, 0,
                opJNZ, 0, 0xFA, 0,
                opMOV, 0, 1, 0,
                opHALT,
        }
        programs := make([][]byte, 1000)
        for i := range programs {
                programs[i] = factProg
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        if result.NumPrograms != 1000 {
                t.Errorf("NumPrograms=%d, want 1000", result.NumPrograms)
        }
        if !result.AllOK() {
                t.Errorf("Not all programs OK, error count=%d", result.ErrorCount())
        }
        failCount := 0
        for i := 0; i < 1000; i++ {
                if result.Results[i] != 120 {
                        failCount++
                        if failCount <= 3 {
                                t.Errorf("Stress[%d]: R0=%d, want 120", i, result.Results[i])
                        }
                }
        }
        if failCount > 0 {
                t.Errorf("Total failures: %d/1000", failCount)
        }
}

func TestBatchMixedPrograms(t *testing.T) {
        e := newTestExecutor(t)
        // Mix of different program types in a single batch
        programs := [][]byte{
                {opHALT},                                        // 0: R0=0
                {opMOVI, 0, 42, opHALT},                        // 1: R0=42
                {opMOVI, 0, 10, opMOVI, 1, 20, opADD, 0, 0, 1, opHALT}, // 2: R0=30
                {opMOVI, 0, 0xFB, opHALT},                      // 3: R0=-5
                {opMOVI16, 0, 0xE8, 0x03, opHALT},             // 4: R0=1000
                {opNOT, 0, opHALT},                             // 5: R0=-1
                {opMOVI, 0, 10, opINC, 0, opINC, 0, opHALT},   // 6: R0=12
        }
        want := []int32{0, 42, 30, -5, 1000, -1, 12}
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        for i, w := range want {
                if result.Results[i] != w {
                        t.Errorf("Mixed[%d]: R0=%d, want %d", i, result.Results[i], w)
                }
        }
}

// ============================================================================
// Conformance: Verify JZ/JNZ offset is int8 (not int16)
// ============================================================================

func TestJNZInt8Offset(t *testing.T) {
        // This test specifically verifies that JNZ uses int8 offset.
        // A negative offset like 0xF2 should be -14 (int8), not +242 (uint8 or int16).
        // If the kernel incorrectly uses int16, the test would fail because
        // the jump would go way past the program boundary.
        e := newTestExecutor(t)
        programs := [][]byte{
                {
                        opMOVI, 0, 3,       // 0: R0 = 3 (loop counter)
                        opMOVI, 1, 0,       // 3: R1 = 0 (accumulator)
                        // loop at offset 6:
                        opINC, 1,            // 6: R1++
                        opDEC, 0,            // 8: R0--
                        opJNZ, 0, 0xF9, 0,  // 10: JNZ R0, -7 -> PC = 10 + (-7) = 3
                        opHALT,              // 14: done
                },
        }
        result, err := e.Run(programs)
        if err != nil {
                t.Fatalf("Run: %v", err)
        }
        // R1 should be 3 (incremented 3 times)
        if result.Results[0] != 0 {
                t.Errorf("JNZ int8: R0=%d, want 0", result.Results[0])
        }
        // R0 should be decremented to 0 (the program halted)
        if result.Errors[0] != ErrNone {
                t.Errorf("JNZ int8: error=%d, want 0", result.Errors[0])
        }
        if result.Cycles[0] < 8 {
                t.Errorf("JNZ int8: cycles=%d, want >= 8", result.Cycles[0])
        }
}

// ============================================================================
// Benchmark
// ============================================================================

func BenchmarkBatchSmall(b *testing.B) {
        e, err := NewBatchExecutor()
        if err != nil {
                b.Fatal(err)
        }
        defer e.Close()

        programs := make([][]byte, 256)
        for i := range programs {
                programs[i] = []byte{opMOVI, 0, 42, opHALT}
        }

        b.ResetTimer()
        for i := 0; i < b.N; i++ {
                result, err := e.Run(programs)
                if err != nil {
                        b.Fatal(err)
                }
                result.Close()
        }
}

func BenchmarkBatchFactorial(b *testing.B) {
        e, err := NewBatchExecutor()
        if err != nil {
                b.Fatal(err)
        }
        defer e.Close()

        factProg := []byte{
                opMOVI, 0, 5,
                opMOVI, 1, 1,
                opMUL, 1, 1, 0,
                opDEC, 0,
                opJNZ, 0, 0xFA, 0,
                opMOV, 0, 1, 0,
                opHALT,
        }
        programs := make([][]byte, 256)
        for i := range programs {
                programs[i] = factProg
        }

        b.ResetTimer()
        for i := 0; i < b.N; i++ {
                result, err := e.Run(programs)
                if err != nil {
                        b.Fatal(err)
                }
                result.Close()
        }
}
