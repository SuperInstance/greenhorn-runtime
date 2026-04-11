#ifndef FLUX_VM_HPP
#define FLUX_VM_HPP

#include <cstdint>
#include <array>
#include <vector>

// FLUX Unified ISA opcodes
constexpr uint8_t OP_HALT = 0x00;
constexpr uint8_t OP_NOP  = 0x01;
constexpr uint8_t OP_INC  = 0x08;
constexpr uint8_t OP_DEC  = 0x09;
constexpr uint8_t OP_NOT  = 0x0A;
constexpr uint8_t OP_NEG  = 0x0B;
constexpr uint8_t OP_PUSH = 0x0C;
constexpr uint8_t OP_POP  = 0x0D;
constexpr uint8_t OP_STRIPCONF = 0x17;
constexpr uint8_t OP_MOVI = 0x18;
constexpr uint8_t OP_ADDI = 0x19;
constexpr uint8_t OP_SUBI = 0x1A;
constexpr uint8_t OP_ADD  = 0x20;
constexpr uint8_t OP_SUB  = 0x21;
constexpr uint8_t OP_MUL  = 0x22;
constexpr uint8_t OP_DIV  = 0x23;
constexpr uint8_t OP_MOD  = 0x24;
constexpr uint8_t OP_AND  = 0x25;
constexpr uint8_t OP_OR   = 0x26;
constexpr uint8_t OP_XOR  = 0x27;
constexpr uint8_t OP_SHL  = 0x28;
constexpr uint8_t OP_SHR  = 0x29;
constexpr uint8_t OP_MIN  = 0x2A;
constexpr uint8_t OP_MAX  = 0x2B;
constexpr uint8_t OP_CMP_EQ = 0x2C;
constexpr uint8_t OP_CMP_LT = 0x2D;
constexpr uint8_t OP_CMP_GT = 0x2E;
constexpr uint8_t OP_CMP_NE = 0x2F;
constexpr uint8_t OP_MOV  = 0x3A;
constexpr uint8_t OP_JZ   = 0x3C;
constexpr uint8_t OP_JNZ  = 0x3D;
constexpr uint8_t OP_MOVI16 = 0x40;
constexpr uint8_t OP_JMP  = 0x43;
constexpr uint8_t OP_LOOP = 0x46;

constexpr int NUM_REGS = 64;
constexpr int STACK_SIZE = 4096;
constexpr int MEM_SIZE = 65536;

class FluxVM {
public:
    std::array<int32_t, NUM_REGS> gp{};
    std::array<int32_t, NUM_REGS> conf{};
    std::array<int32_t, STACK_SIZE> stack{};
    std::array<uint8_t, MEM_SIZE> memory{};
    int32_t sp = STACK_SIZE;
    int32_t pc = 0;
    bool halted = false;
    int cycles = 0;
    int strip_n = 0;

    FluxVM() { gp.fill(0); conf.fill(0); stack.fill(0); memory.fill(0); }
    
    int execute(const std::vector<uint8_t>& bc);
    int step(const std::vector<uint8_t>& bc);
    static int formatSize(uint8_t op);
};

#endif
