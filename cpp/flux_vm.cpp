#include "flux_vm.hpp"
#include <algorithm>
#include <cstring>

int FluxVM::formatSize(uint8_t op) {
    if (op <= 0x07) return 1;
    if (op <= 0x0F) return 2;
    if (op <= 0x17) return 2;
    if (op <= 0x1F) return 3;
    if (op <= 0x4F) return 4;
    return 1;
}

int FluxVM::execute(const std::vector<uint8_t>& bc) {
    while (!halted && pc < (int)bc.size()) {
        int r = step(bc);
        if (r < 0) break;
        if (r > 0) pc += r;
        // r == 0: jump already set pc
    }
    return cycles;
}

int FluxVM::step(const std::vector<uint8_t>& bc) {
    if (pc >= (int)bc.size()) return -1;
    uint8_t op = bc[pc];
    int sz = formatSize(op);
    if (pc + sz > (int)bc.size()) { halted = true; return -1; }
    cycles++;
    if (strip_n > 0) strip_n--;

    switch (op) {
        case OP_HALT: halted = true; return 1;
        case OP_NOP: return 1;
        case OP_INC: gp[bc[pc+1]]++; return 2;
        case OP_DEC: gp[bc[pc+1]]--; return 2;
        case OP_NOT: gp[bc[pc+1]] = ~gp[bc[pc+1]]; return 2;
        case OP_NEG: gp[bc[pc+1]] = -gp[bc[pc+1]]; return 2;
        case OP_PUSH: stack[--sp] = gp[bc[pc+1]]; return 2;
        case OP_POP: gp[bc[pc+1]] = stack[sp++]; return 2;
        case OP_STRIPCONF: strip_n = bc[pc+1]; return 2;
        case OP_MOVI: gp[bc[pc+1]] = (int32_t)(int8_t)bc[pc+2]; return 3;
        case OP_ADDI: gp[bc[pc+1]] += (int32_t)(int8_t)bc[pc+2]; return 3;
        case OP_SUBI: gp[bc[pc+1]] -= (int32_t)(int8_t)bc[pc+2]; return 3;
        case OP_ADD: gp[bc[pc+1]] = gp[bc[pc+2]] + gp[bc[pc+3]]; return 4;
        case OP_SUB: gp[bc[pc+1]] = gp[bc[pc+2]] - gp[bc[pc+3]]; return 4;
        case OP_MUL: gp[bc[pc+1]] = gp[bc[pc+2]] * gp[bc[pc+3]]; return 4;
        case OP_DIV: 
            if (gp[bc[pc+3]] == 0) { halted = true; return 4; }
            gp[bc[pc+1]] = gp[bc[pc+2]] / gp[bc[pc+3]]; return 4;
        case OP_MOD: gp[bc[pc+1]] = gp[bc[pc+2]] % gp[bc[pc+3]]; return 4;
        case OP_AND: gp[bc[pc+1]] = gp[bc[pc+2]] & gp[bc[pc+3]]; return 4;
        case OP_OR:  gp[bc[pc+1]] = gp[bc[pc+2]] | gp[bc[pc+3]]; return 4;
        case OP_XOR: gp[bc[pc+1]] = gp[bc[pc+2]] ^ gp[bc[pc+3]]; return 4;
        case OP_MOV: gp[bc[pc+1]] = gp[bc[pc+2]]; return 4;
        case OP_CMP_EQ: gp[bc[pc+1]] = (gp[bc[pc+2]] == gp[bc[pc+3]]) ? 1 : 0; return 4;
        case OP_CMP_LT: gp[bc[pc+1]] = (gp[bc[pc+2]] < gp[bc[pc+3]]) ? 1 : 0; return 4;
        case OP_CMP_GT: gp[bc[pc+1]] = (gp[bc[pc+2]] > gp[bc[pc+3]]) ? 1 : 0; return 4;
        case OP_CMP_NE: gp[bc[pc+1]] = (gp[bc[pc+2]] != gp[bc[pc+3]]) ? 1 : 0; return 4;
        case OP_MIN: gp[bc[pc+1]] = std::min(gp[bc[pc+2]], gp[bc[pc+3]]); return 4;
        case OP_MAX: gp[bc[pc+1]] = std::max(gp[bc[pc+2]], gp[bc[pc+3]]); return 4;
        case OP_JNZ: {
            uint8_t rd = bc[pc+1];
            int32_t off = (int32_t)(int8_t)bc[pc+2];
            if (gp[rd] != 0) { pc += off; return 0; }
            return 4;
        }
        case OP_JZ: {
            uint8_t rd = bc[pc+1];
            int32_t off = (int32_t)(int8_t)bc[pc+2];
            if (gp[rd] == 0) { pc += off; return 0; }
            return 4;
        }
        case OP_MOVI16: {
            uint8_t rd = bc[pc+1];
            int16_t imm = (int16_t)((uint16_t)bc[pc+2] | ((uint16_t)bc[pc+3] << 8));
            gp[rd] = (int32_t)imm;
            return 4;
        }
        case OP_JMP: {
            int16_t off = (int16_t)((uint16_t)bc[pc+2] | ((uint16_t)bc[pc+3] << 8));
            pc += (int32_t)off;
            return 0;
        }
        case OP_LOOP: {
            uint8_t rd = bc[pc+1];
            uint16_t off = (uint16_t)bc[pc+2] | ((uint16_t)bc[pc+3] << 8);
            gp[rd]--;
            if (gp[rd] > 0) { pc -= (int32_t)off; return 0; }
            return 4;
        }
        default: return sz;
    }
}
