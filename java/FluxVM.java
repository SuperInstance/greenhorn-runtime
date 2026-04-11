/**
 * FLUX Unified ISA — Java Implementation
 * Matches Go/C/C++/Zig/Rust runtimes exactly.
 */
public class FluxVM {
    public static final int NUM_REGS = 64;
    public static final int STACK_SIZE = 4096;
    public static final int MEM_SIZE = 65536;

    public int[] gp = new int[NUM_REGS];
    public int[] conf = new int[NUM_REGS];
    public int[] stack = new int[STACK_SIZE];
    public byte[] memory = new byte[MEM_SIZE];
    public int sp = STACK_SIZE;
    public int pc = 0;
    public boolean halted = false;
    public int cycles = 0;
    public int stripN = 0;

    public int execute(byte[] bc) {
        while (!halted && pc >= 0 && pc < bc.length) {
            int r = step(bc);
            if (r < 0) break;
            if (r > 0) pc += r;
        }
        return cycles;
    }

    static int formatSize(int op) {
        if (op <= 0x07) return 1;
        if (op <= 0x17) return 2;
        if (op <= 0x1F) return 3;
        if (op <= 0x4F) return 4;
        return 1;
    }

    int step(byte[] bc) {
        if (pc >= bc.length) return -1;
        int op = bc[pc] & 0xFF;
        int sz = formatSize(op);
        if (pc + sz > bc.length) { halted = true; return -1; }
        cycles++;
        if (stripN > 0) stripN--;

        switch (op) {
            case 0x00: halted = true; return 1; // HALT
            case 0x01: return 1; // NOP
            case 0x08: gp[bc[pc+1] & 0x3F]++; return 2; // INC
            case 0x09: gp[bc[pc+1] & 0x3F]--; return 2; // DEC
            case 0x0A: { int rd = bc[pc+1] & 0x3F; gp[rd] = ~gp[rd]; return 2; } // NOT
            case 0x0B: { int rd = bc[pc+1] & 0x3F; gp[rd] = -gp[rd]; return 2; } // NEG
            case 0x0C: stack[--sp] = gp[bc[pc+1] & 0x3F]; return 2; // PUSH
            case 0x0D: { int rd = bc[pc+1] & 0x3F; gp[rd] = stack[sp++]; return 2; } // POP
            case 0x17: stripN = bc[pc+1] & 0xFF; return 2; // STRIPCONF
            case 0x18: { gp[bc[pc+1] & 0x3F] = bc[pc+2]; return 3; } // MOVI
            case 0x19: { gp[bc[pc+1] & 0x3F] += bc[pc+2]; return 3; } // ADDI
            case 0x1A: { gp[bc[pc+1] & 0x3F] -= bc[pc+2]; return 3; } // SUBI
            case 0x20: { gp[bc[pc+1]&0x3F] = gp[bc[pc+2]&0x3F] + gp[bc[pc+3]&0x3F]; return 4; }
            case 0x21: { gp[bc[pc+1]&0x3F] = gp[bc[pc+2]&0x3F] - gp[bc[pc+3]&0x3F]; return 4; }
            case 0x22: { gp[bc[pc+1]&0x3F] = gp[bc[pc+2]&0x3F] * gp[bc[pc+3]&0x3F]; return 4; }
            case 0x23: { int d = gp[bc[pc+3]&0x3F]; if(d==0){halted=true;return 4;} gp[bc[pc+1]&0x3F] = gp[bc[pc+2]&0x3F] / d; return 4; }
            case 0x24: { gp[bc[pc+1]&0x3F] = gp[bc[pc+2]&0x3F] % gp[bc[pc+3]&0x3F]; return 4; }
            case 0x25: { gp[bc[pc+1]&0x3F] = gp[bc[pc+2]&0x3F] & gp[bc[pc+3]&0x3F]; return 4; }
            case 0x26: { gp[bc[pc+1]&0x3F] = gp[bc[pc+2]&0x3F] | gp[bc[pc+3]&0x3F]; return 4; }
            case 0x27: { gp[bc[pc+1]&0x3F] = gp[bc[pc+2]&0x3F] ^ gp[bc[pc+3]&0x3F]; return 4; }
            case 0x28: { gp[bc[pc+1]&0x3F] = gp[bc[pc+2]&0x3F] << gp[bc[pc+3]&0x3F]; return 4; }
            case 0x29: { gp[bc[pc+1]&0x3F] = gp[bc[pc+2]&0x3F] >>> gp[bc[pc+3]&0x3F]; return 4; }
            case 0x2A: { int a=gp[bc[pc+2]&0x3F],b=gp[bc[pc+3]&0x3F]; gp[bc[pc+1]&0x3F]=Math.min(a,b); return 4; }
            case 0x2B: { int a=gp[bc[pc+2]&0x3F],b=gp[bc[pc+3]&0x3F]; gp[bc[pc+1]&0x3F]=Math.max(a,b); return 4; }
            case 0x2C: { gp[bc[pc+1]&0x3F] = (gp[bc[pc+2]&0x3F] == gp[bc[pc+3]&0x3F]) ? 1 : 0; return 4; }
            case 0x2D: { gp[bc[pc+1]&0x3F] = (gp[bc[pc+2]&0x3F] < gp[bc[pc+3]&0x3F]) ? 1 : 0; return 4; }
            case 0x2E: { gp[bc[pc+1]&0x3F] = (gp[bc[pc+2]&0x3F] > gp[bc[pc+3]&0x3F]) ? 1 : 0; return 4; }
            case 0x2F: { gp[bc[pc+1]&0x3F] = (gp[bc[pc+2]&0x3F] != gp[bc[pc+3]&0x3F]) ? 1 : 0; return 4; }
            case 0x3A: { gp[bc[pc+1]&0x3F] = gp[bc[pc+2]&0x3F]; return 4; } // MOV
            case 0x3C: { int rd=bc[pc+1]&0x3F; int off=bc[pc+2]; if(gp[rd]==0){pc+=off;return 0;} return 4; }
            case 0x3D: { int rd=bc[pc+1]&0x3F; int off=bc[pc+2]; if(gp[rd]!=0){pc+=off;return 0;} return 4; }
            case 0x40: { // MOVI16
                int rd = bc[pc+1] & 0x3F;
                int imm = ((bc[pc+2] & 0xFF)) | ((bc[pc+3] & 0xFF) << 8);
                if ((imm & 0x8000) != 0) imm |= 0xFFFF0000;
                gp[rd] = imm;
                return 4;
            }
            case 0x43: { int off = ((bc[pc+2]&0xFF)) | ((bc[pc+3]&0xFF)<<8); if((off&0x8000)!=0)off|=0xFFFF0000; pc += off; return 0; }
            case 0x46: { // LOOP
                int rd = bc[pc+1] & 0x3F;
                int off = ((bc[pc+2]&0xFF)) | ((bc[pc+3]&0xFF)<<8);
                gp[rd]--;
                if (gp[rd] > 0) { pc -= off; return 0; }
                return 4;
            }
            default: return sz;
        }
    }
}
