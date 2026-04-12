package vm

import (
	"testing"
)

func TestHalt(t *testing.T) {
	v := New([]byte{0x00}) // HALT
	err := v.Execute()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !v.Halted {
		t.Error("expected halted")
	}
	if v.Cycles != 1 {
		t.Errorf("expected 1 cycle, got %d", v.Cycles)
	}
}

func TestNop(t *testing.T) {
	v := New([]byte{0x01, 0x01, 0x01, 0x00}) // NOP NOP NOP HALT
	err := v.Execute()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if v.Cycles != 4 {
		t.Errorf("expected 4 cycles, got %d", v.Cycles)
	}
}

func TestMoviPositive(t *testing.T) {
	v := New([]byte{0x18, 0x01, 0x2A, 0x00, 0x00}) // MOVI R1, 42; HALT
	err := v.Execute()
	if err != nil { t.Fatal(err) }
	if v.Registers[1] != 42 { t.Errorf("R1 = %d, want 42", v.Registers[1]) }
}

func TestMoviNegative(t *testing.T) {
	// -1 = 0xFFFF as int16
	v := New([]byte{0x18, 0x01, 0xFF, 0xFF, 0x00}) // MOVI R1, -1; HALT
	err := v.Execute()
	if err != nil { t.Fatal(err) }
	if v.Registers[1] != -1 { t.Errorf("R1 = %d, want -1", v.Registers[1]) }
}

func TestMoviZero(t *testing.T) {
	v := New([]byte{0x18, 0x01, 0x00, 0x00, 0x00}) // MOVI R1, 0; HALT
	err := v.Execute()
	if err != nil { t.Fatal(err) }
	if v.Registers[1] != 0 { t.Errorf("R1 = %d, want 0", v.Registers[1]) }
}

func TestAdd(t *testing.T) {
	// MOVI R1, 10; MOVI R2, 20; ADD R3, R1, R2; HALT
	bc := append(EncodeMOVI(1, 10), EncodeMOVI(2, 20)...)
	bc = append(bc, EncodeE(0x20, 3, 1, 2)...)
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[3] != 30 { t.Errorf("R3 = %d, want 30", v.Registers[3]) }
}

func TestSub(t *testing.T) {
	bc := append(EncodeMOVI(1, 50), EncodeMOVI(2, 17)...)
	bc = append(bc, EncodeE(0x21, 3, 1, 2)...)
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[3] != 33 { t.Errorf("R3 = %d, want 33", v.Registers[3]) }
}

func TestMul(t *testing.T) {
	bc := append(EncodeMOVI(1, 6), EncodeMOVI(2, 7)...)
	bc = append(bc, EncodeE(0x22, 3, 1, 2)...)
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[3] != 42 { t.Errorf("R3 = %d, want 42", v.Registers[3]) }
}

func TestDiv(t *testing.T) {
	bc := append(EncodeMOVI(1, 100), EncodeMOVI(2, 7)...)
	bc = append(bc, EncodeE(0x23, 3, 1, 2)...)
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[3] != 14 { t.Errorf("R3 = %d, want 14", v.Registers[3]) }
}

func TestDivNegative(t *testing.T) {
	bc := append(EncodeMOVI(1, -100), EncodeMOVI(2, 7)...)
	bc = append(bc, EncodeE(0x23, 3, 1, 2)...)
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[3] != -14 { t.Errorf("R3 = %d, want -14", v.Registers[3]) }
}

func TestDivZero(t *testing.T) {
	bc := append(EncodeMOVI(1, 10), EncodeMOVI(2, 0)...)
	bc = append(bc, EncodeE(0x23, 3, 1, 2)...)
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != ErrDivisionByZero { t.Errorf("expected DivZero, got %v", err) }
}

func TestMod(t *testing.T) {
	bc := append(EncodeMOVI(1, 17), EncodeMOVI(2, 5)...)
	bc = append(bc, EncodeE(0x24, 3, 1, 2)...)
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[3] != 2 { t.Errorf("R3 = %d, want 2", v.Registers[3]) }
}

func TestModZero(t *testing.T) {
	bc := append(EncodeMOVI(1, 10), EncodeMOVI(2, 0)...)
	bc = append(bc, EncodeE(0x24, 3, 1, 2)...)
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != ErrDivisionByZero { t.Errorf("expected DivZero, got %v", err) }
}

func TestIncDec(t *testing.T) {
	bc := EncodeMOVI(1, 10)
	bc = append(bc, EncodeB(0x08, 1)...) // INC R1
	bc = append(bc, EncodeB(0x08, 1)...) // INC R1
	bc = append(bc, EncodeB(0x09, 1)...) // DEC R1
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[1] != 11 { t.Errorf("R1 = %d, want 11", v.Registers[1]) }
}

func TestNeg(t *testing.T) {
	bc := EncodeMOVI(1, 42)
	bc = append(bc, EncodeB(0x0B, 1)...) // NEG R1
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[1] != -42 { t.Errorf("R1 = %d, want -42", v.Registers[1]) }
}

func TestNot(t *testing.T) {
	bc := EncodeMOVI(1, 0) // all zeros
	bc = append(bc, EncodeB(0x0A, 1)...) // NOT R1
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[1] != -1 { t.Errorf("R1 = %d, want -1 (all bits set)", v.Registers[1]) }
}

func TestBitwise(t *testing.T) {
	// R1 = 0xFF, R2 = 0x0F, R3 = R1 AND R2 = 0x0F
	bc := EncodeMOVI(1, 0xFF)
	bc = append(bc, EncodeMOVI(2, 0x0F)...)
	bc = append(bc, EncodeE(0x25, 3, 1, 2)...) // AND
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[3] != 0x0F { t.Errorf("R3 = 0x%X, want 0x0F", v.Registers[3]) }
}

func TestOrXor(t *testing.T) {
	bc := EncodeMOVI(1, 0xF0)
	bc = append(bc, EncodeMOVI(2, 0x0F)...)
	bc = append(bc, EncodeE(0x26, 3, 1, 2)...) // OR -> 0xFF
	bc = append(bc, EncodeE(0x27, 4, 1, 2)...) // XOR -> 0xFF
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[3] != 0xFF { t.Errorf("R3 = 0x%X, want 0xFF", v.Registers[3]) }
	if v.Registers[4] != 0xFF { t.Errorf("R4 = 0x%X, want 0xFF", v.Registers[4]) }
}

func TestCmpEq(t *testing.T) {
	bc := EncodeMOVI(1, 42)
	bc = append(bc, EncodeMOVI(2, 42)...)
	bc = append(bc, EncodeE(0x2C, 3, 1, 2)...) // CMP_EQ R3, R1, R2
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[3] != 1 { t.Errorf("R3 = %d, want 1 (equal)", v.Registers[3]) }
	if !v.Flags.Zero { t.Error("expected Zero flag set") }
}

func TestCmpEqFalse(t *testing.T) {
	bc := EncodeMOVI(1, 10)
	bc = append(bc, EncodeMOVI(2, 20)...)
	bc = append(bc, EncodeE(0x2C, 3, 1, 2)...)
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[3] != 0 { t.Errorf("R3 = %d, want 0", v.Registers[3]) }
}

func TestCmpLtGt(t *testing.T) {
	bc := EncodeMOVI(1, 10)
	bc = append(bc, EncodeMOVI(2, 20)...)
	bc = append(bc, EncodeE(0x2D, 3, 1, 2)...) // CMP_LT R3 = 1
	bc = append(bc, EncodeE(0x2E, 4, 1, 2)...) // CMP_GT R4 = 0
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[3] != 1 { t.Errorf("R3 = %d, want 1 (lt)", v.Registers[3]) }
	if v.Registers[4] != 0 { t.Errorf("R4 = %d, want 0 (not gt)", v.Registers[4]) }
}

func TestCmpNe(t *testing.T) {
	bc := EncodeMOVI(1, 10)
	bc = append(bc, EncodeMOVI(2, 20)...)
	bc = append(bc, EncodeE(0x2F, 3, 1, 2)...)
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[3] != 1 { t.Errorf("R3 = %d, want 1 (ne)", v.Registers[3]) }
}

func TestJmpForward(t *testing.T) {
	// JMP +4 (skip over INC R1), INC R2, HALT
	// JMP is at PC=0, size=4, so target=0+4+4=8
	bc := EncodeJMP(4)       // PC=0: JMP +4 -> PC=8
	bc = append(bc, EncodeB(0x08, 1)...) // PC=4: INC R1 (skipped)
	bc = append(bc, EncodeB(0x08, 2)...) // PC=6: INC R2 (skipped)
	bc = append(bc, 0x00)                // PC=8: HALT
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[1] != 0 { t.Errorf("R1 should be 0 (skipped), got %d", v.Registers[1]) }
	if v.Registers[2] != 0 { t.Errorf("R2 should be 0 (skipped), got %d", v.Registers[2]) }
}

func TestJzTaken(t *testing.T) {
	// R1=0; JZ R1, +4; INC R1; HALT
	bc := EncodeMOVI(1, 0)
	bc = append(bc, EncodeBranch(0x44, 1, 4)...) // JZ R1, +4
	bc = append(bc, EncodeB(0x08, 1)...)          // INC R1 (skipped)
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[1] != 0 { t.Errorf("R1 should be 0 (skipped), got %d", v.Registers[1]) }
}

func TestJzNotTaken(t *testing.T) {
	// R1=1; JZ R1, +4; INC R1; HALT
	bc := EncodeMOVI(1, 1)
	bc = append(bc, EncodeBranch(0x44, 1, 4)...) // JZ R1, +4 (not taken)
	bc = append(bc, EncodeB(0x08, 1)...)          // INC R1
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[1] != 2 { t.Errorf("R1 = %d, want 2", v.Registers[1]) }
}

func TestJnz(t *testing.T) {
	// R1=1; JNZ R1, +4; INC R1; HALT (should skip)
	bc := EncodeMOVI(1, 1)
	bc = append(bc, EncodeBranch(0x45, 1, 4)...)
	bc = append(bc, EncodeB(0x08, 1)...)
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[1] != 1 { t.Errorf("R1 = %d, want 1 (skipped INC)", v.Registers[1]) }
}

func TestCallRet(t *testing.T) {
	// MOVI R1, 10
	// CALL +4 (skip next 4 bytes, push ret addr)
	// INC R1 (skipped)
	// HALT (skipped)
	// INC R2 (at call target)
	// RET
	// INC R3 (after return)
	// HALT
	bc := EncodeMOVI(1, 10)              // 0-3
	bc = append(bc, EncodeCALL(4)...)     // 4-7: CALL +4 -> target=8+4=12
	bc = append(bc, EncodeB(0x08, 1)...) // 8-9: INC R1 (skipped)
	bc = append(bc, 0x00)                // 10: HALT (skipped)
	bc = append(bc, EncodeB(0x08, 2)...) // 11-12: this is actually 11... let me recalculate
	// Actually: CALL at PC=4, instruction size=4, ret_addr=8
	// CALL offset=4, target = 8+4 = 12
	bc = make([]byte, 0)
	bc = append(bc, EncodeMOVI(1, 10)...)    // 0-3
	bc = append(bc, EncodeCALL(8)...)        // 4-7: CALL +8, ret_addr=8, target=8+8=16
	bc = append(bc, 0x00)                   // 8: HALT (not reached by main flow)
	bc = append(bc, EncodeB(0x08, 1)...)    // 9-10: padding
	bc = append(bc, EncodeB(0x08, 1)...)    // 11-12: padding  
	bc = append(bc, 0x00)                   // 13: padding
	bc = append(bc, EncodeB(0x08, 2)...)    // 14-15: more padding
	bc = append(bc, EncodeB(0x08, 2)...)    // 16-17: INC R2 (call target)
	bc = append(bc, 0x02)                   // 18: RET -> back to PC=8
	bc = append(bc, EncodeB(0x08, 3)...)    // 19-20: INC R3
	bc = append(bc, 0x00)                   // 21: HALT
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[1] != 10 { t.Errorf("R1 = %d, want 10", v.Registers[1]) }
	if v.Registers[2] != 1 { t.Errorf("R2 = %d, want 1", v.Registers[2]) }
	if v.Registers[3] != 1 { t.Errorf("R3 = %d, want 1", v.Registers[3]) }
}

func TestPushPop(t *testing.T) {
	bc := EncodeMOVI(1, 42)
	bc = append(bc, EncodeB(0x0C, 1)...) // PUSH R1
	bc = append(bc, EncodeB(0x09, 1)...) // DEC R1
	bc = append(bc, EncodeB(0x0D, 1)...) // POP R1
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[1] != 42 { t.Errorf("R1 = %d, want 42 (restored from stack)", v.Registers[1]) }
}

func TestPopEmptyStack(t *testing.T) {
	bc := EncodeB(0x0D, 1) // POP R1 (empty stack)
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != ErrStackUnderflow { t.Errorf("expected StackUnderflow, got %v", err) }
}

func TestR0Immutable(t *testing.T) {
	bc := EncodeMOVI(0, 42) // MOVI R0, 42 (should be ignored)
	bc = append(bc, EncodeB(0x08, 0)...) // INC R0 (should be ignored)
	bc = append(bc, EncodeE(0x20, 0, 0, 0)...) // ADD R0, R0, R0 (ignored)
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[0] != 0 { t.Errorf("R0 = %d, want 0 (immutable)", v.Registers[0]) }
}

func TestAgentStubs(t *testing.T) {
	for _, op := range []byte{0x50, 0x51, 0x53} {
		v := New([]byte{op})
		if err := v.Execute(); err != ErrStub {
			t.Errorf("opcode 0x%02x: expected ErrStub, got %v", op, err)
		}
	}
}

func TestCycleLimit(t *testing.T) {
	// Infinite loop: JMP -4 (back to self)
	bc := EncodeJMP(-4)
	v := New(bc)
	v.MaxCycles = 100
	if err := v.Execute(); err != ErrCycleLimit { t.Errorf("expected CycleLimit, got %v", err) }
}

func TestAddiSubi(t *testing.T) {
	bc := EncodeMOVI(1, 100)
	bc = append(bc, EncodeMOVI(1, 100)...) // reset
	bc = append(bc, []byte{0x19, 0x01, 0x0A, 0x00}...) // ADDI R1, 10
	bc = append(bc, []byte{0x1A, 0x01, 0x05, 0x00}...) // SUBI R1, 5
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[1] != 105 { t.Errorf("R1 = %d, want 105", v.Registers[1]) }
}

func TestEndOfBytecode(t *testing.T) {
	v := New([]byte{0x01, 0x01}) // NOP NOP (no HALT)
	err := v.Execute()
	if err != nil { t.Fatalf("expected implicit halt, got %v", err) }
	if !v.Halted { t.Error("expected halted") }
}

func TestInvalidOpcode(t *testing.T) {
	v := New([]byte{0xFE})
	err := v.Execute()
	if err == nil { t.Fatal("expected error for invalid opcode") }
}

func TestArithmeticOverflow(t *testing.T) {
	// Build up large values to test wrapping
	bc := EncodeMOVI(1, 30000)
	bc = append(bc, EncodeMOVI(2, 30000)...)
	bc = append(bc, EncodeE(0x20, 3, 1, 2)...) // ADD -> 60000
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[3] != 60000 { t.Errorf("R3 = %d, want 60000", v.Registers[3]) }
}

func TestLoopCountdown(t *testing.T) {
	// R1 = 5; loop: DEC R1; JNZ R1, -2; HALT
	bc := EncodeMOVI(1, 5)
	bc = append(bc, EncodeB(0x09, 1)...)           // DEC R1
	bc = append(bc, EncodeBranch(0x45, 1, -2)...) // JNZ R1, -2
	bc = append(bc, 0x00)
	v := New(bc)
	if err := v.Execute(); err != nil { t.Fatal(err) }
	if v.Registers[1] != 0 { t.Errorf("R1 = %d, want 0", v.Registers[1]) }
}
