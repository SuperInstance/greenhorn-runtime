# FLUX VM — Go Implementation

The second FLUX bytecode runtime, implementing the **Unified ISA** specification.

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/superinstance/greenhorn-runtime/pkg/vm"
)

func main() {
    // MOVI R1, 42; ADD R2, R1, R1; HALT
    bc := append(vm.EncodeMOVI(1, 42), vm.EncodeE(0x20, 2, 1, 1)...)
    bc = append(bc, 0x00)

    v := vm.New(bc)
    if err := v.Execute(); err != nil {
        panic(err)
    }
    fmt.Printf("R1=%d R2=%d\n", v.Registers[1], v.Registers[2])
    // Output: R1=42 R2=84
}
```

## Architecture

| Component | Description |
|-----------|-------------|
| Registers | 64 × int32, R0 hardwired to zero |
| Memory | 64KB byte-addressable |
| Stack | Dynamic int32 stack, 65536 max depth |
| Flags | Zero, Sign, Carry, Overflow |
| Safety | 10M cycle limit (configurable) |

## Supported Opcodes

### Format A (1 byte)
| Opcode | Hex | Description |
|--------|-----|-------------|
| HALT | 0x00 | Stop execution |
| NOP | 0x01 | No operation |
| RET | 0x02 | Pop PC from stack |

### Format B (2 bytes)
| Opcode | Hex | Description |
|--------|-----|-------------|
| INC | 0x08 | rd += 1 |
| DEC | 0x09 | rd -= 1 |
| NOT | 0x0A | rd = ~rd |
| NEG | 0x0B | rd = -rd |
| PUSH | 0x0C | Push rd to stack |
| POP | 0x0D | Pop stack to rd |
| TELL | 0x50 | Agent tell (stub) |
| ASK | 0x51 | Agent ask (stub) |
| BCAST | 0x53 | Agent broadcast (stub) |

### Format E (4 bytes: op, rd, rs1, rs2)
| Opcode | Hex | Description |
|--------|-----|-------------|
| ADD | 0x20 | rd = rs1 + rs2 |
| SUB | 0x21 | rd = rs1 - rs2 |
| MUL | 0x22 | rd = rs1 * rs2 |
| DIV | 0x23 | rd = rs1 / rs2 |
| MOD | 0x24 | rd = rs1 % rs2 |
| AND | 0x25 | rd = rs1 & rs2 |
| OR | 0x26 | rd = rs1 \| rs2 |
| XOR | 0x27 | rd = rs1 ^ rs2 |
| CMP_EQ | 0x2C | rd = (rs1 == rs2) ? 1 : 0 |
| CMP_LT | 0x2D | rd = (rs1 < rs2) ? 1 : 0 |
| CMP_GT | 0x2E | rd = (rs1 > rs2) ? 1 : 0 |
| CMP_NE | 0x2F | rd = (rs1 != rs2) ? 1 : 0 |

### Format F (4 bytes: op, rd, imm16_lo, imm16_hi)
| Opcode | Hex | Description |
|--------|-----|-------------|
| MOVI | 0x18 | rd = sign_extend(imm16) |
| ADDI | 0x19 | rd += sign_extend(imm16) |
| SUBI | 0x1A | rd -= sign_extend(imm16) |
| JMP | 0x43 | PC += imm16 |
| JZ | 0x44 | if (rd == 0) PC += imm16 |
| JNZ | 0x45 | if (rd != 0) PC += imm16 |
| CALL | 0x4A | push(PC+4); PC += imm16 |

## Relationship to Python VM

The Python VM in `flux-runtime` is the original implementation. This Go VM is the second runtime. Both target the same **Unified ISA** where HALT=0x00 and A2A opcodes are at 0x50-0x5F.

Use the test vectors in `flux-conformance` to verify identical behavior across both runtimes.

## Encoding

All multi-byte values are little-endian. imm16 values are sign-extended to int32. Register fields are 6-bit (0-63).

## Error Handling

The VM returns sentinel errors:
- `ErrHalted` — normal termination
- `ErrCycleLimit` — exceeded MaxCycles
- `ErrDivisionByZero` — divide or modulo by zero
- `ErrStackOverflow` / `ErrStackUnderflow`
- `ErrStub` — agent opcode not yet implemented
- `ErrInvalidOpcode` — unrecognized opcode byte
