const OP = {
  HALT:0x00,NOP:0x01,INC:0x08,DEC:0x09,NOT:0x0A,NEG:0x0B,
  PUSH:0x0C,POP:0x0D,STRIPCONF:0x17,
  MOVI:0x18,ADDI:0x19,SUBI:0x1A,
  ADD:0x20,SUB:0x21,MUL:0x22,DIV:0x23,MOD:0x24,
  AND:0x25,OR:0x26,XOR:0x27,
  MIN:0x2A,MAX:0x2B,
  CMP_EQ:0x2C,CMP_LT:0x2D,CMP_GT:0x2E,CMP_NE:0x2F,
  MOV:0x3A,JZ:0x3C,JNZ:0x3D,
  MOVI16:0x40,JMP:0x43,LOOP:0x46,
};

class FluxVM {
  constructor() {
    this.gp = new Int32Array(64);
    this.conf = new Int32Array(64);
    this.stack = new Int32Array(4096);
    this.sp = 4096;
    this.pc = 0;
    this.halted = false;
    this.cycles = 0;
    this.stripN = 0;
  }
  execute(bc) {
    while (!this.halted && this.pc >= 0 && this.pc < bc.length) {
      const r = this.step(bc);
      if (r < 0) break;
      if (r > 0) this.pc += r;
    }
    return this.cycles;
  }
  static fmtSz(op) {
    if (op <= 0x07) return 1;
    if (op <= 0x17) return 2;
    if (op <= 0x1F) return 3;
    if (op <= 0x4F) return 4;
    return 1;
  }
  step(bc) {
    if (this.pc >= bc.length) return -1;
    const op = bc[this.pc];
    const sz = FluxVM.fmtSz(op);
    if (this.pc + sz > bc.length) { this.halted = true; return -1; }
    this.cycles++;
    if (this.stripN > 0) this.stripN--;
    const rd = (i) => bc[this.pc + i];
    switch (op) {
      case OP.HALT: this.halted = true; return 1;
      case OP.NOP: return 1;
      case OP.INC: this.gp[rd(1)]++; return 2;
      case OP.DEC: this.gp[rd(1)]--; return 2;
      case OP.NOT: this.gp[rd(1)] = ~this.gp[rd(1)]; return 2;
      case OP.NEG: this.gp[rd(1)] = -this.gp[rd(1)]; return 2;
      case OP.PUSH: this.stack[--this.sp] = this.gp[rd(1)]; return 2;
      case OP.POP: this.gp[rd(1)] = this.stack[this.sp++]; return 2;
      case OP.STRIPCONF: this.stripN = rd(1); return 2;
      case OP.MOVI: this.gp[rd(1)] = (bc[this.pc+2] << 24) >> 24; return 3;
      case OP.ADDI: this.gp[rd(1)] += (bc[this.pc+2] << 24) >> 24; return 3;
      case OP.SUBI: this.gp[rd(1)] -= (bc[this.pc+2] << 24) >> 24; return 3;
      case OP.ADD: this.gp[rd(1)] = this.gp[rd(2)] + this.gp[rd(3)]; return 4;
      case OP.SUB: this.gp[rd(1)] = this.gp[rd(2)] - this.gp[rd(3)]; return 4;
      case OP.MUL: this.gp[rd(1)] = this.gp[rd(2)] * this.gp[rd(3)]; return 4;
      case OP.DIV:
        if (this.gp[rd(3)] === 0) { this.halted = true; return 4; }
        this.gp[rd(1)] = Math.trunc(this.gp[rd(2)] / this.gp[rd(3)]); return 4;
      case OP.MOD: this.gp[rd(1)] = this.gp[rd(2)] % this.gp[rd(3)]; return 4;
      case OP.AND: this.gp[rd(1)] = this.gp[rd(2)] & this.gp[rd(3)]; return 4;
      case OP.OR:  this.gp[rd(1)] = this.gp[rd(2)] | this.gp[rd(3)]; return 4;
      case OP.XOR: this.gp[rd(1)] = this.gp[rd(2)] ^ this.gp[rd(3)]; return 4;
      case OP.MIN: this.gp[rd(1)] = Math.min(this.gp[rd(2)], this.gp[rd(3)]); return 4;
      case OP.MAX: this.gp[rd(1)] = Math.max(this.gp[rd(2)], this.gp[rd(3)]); return 4;
      case OP.CMP_EQ: this.gp[rd(1)] = (this.gp[rd(2)] === this.gp[rd(3)]) ? 1 : 0; return 4;
      case OP.CMP_LT: this.gp[rd(1)] = (this.gp[rd(2)] < this.gp[rd(3)]) ? 1 : 0; return 4;
      case OP.CMP_GT: this.gp[rd(1)] = (this.gp[rd(2)] > this.gp[rd(3)]) ? 1 : 0; return 4;
      case OP.CMP_NE: this.gp[rd(1)] = (this.gp[rd(2)] !== this.gp[rd(3)]) ? 1 : 0; return 4;
      case OP.MOV: this.gp[rd(1)] = this.gp[rd(2)]; return 4;
      case OP.JZ:  if (this.gp[rd(1)] === 0) { this.pc += ((bc[this.pc+2] << 24) >> 24); return 0; } return 4;
      case OP.JNZ: if (this.gp[rd(1)] !== 0) { this.pc += ((bc[this.pc+2] << 24) >> 24); return 0; } return 4;
      case OP.MOVI16: {
        const imm = bc[this.pc+2] | (bc[this.pc+3] << 8);
        this.gp[rd(1)] = imm > 0x7FFF ? imm - 0x10000 : imm;
        return 4;
      }
      case OP.JMP: {
        let off = bc[this.pc+2] | (bc[this.pc+3] << 8);
        if (off > 0x7FFF) off -= 0x10000;
        this.pc += off; return 0;
      }
      case OP.LOOP: {
        const r = rd(1);
        const off = bc[this.pc+2] | (bc[this.pc+3] << 8);
        this.gp[r]--;
        if (this.gp[r] > 0) { this.pc -= off; return 0; }
        return 4;
      }
      default: return sz;
    }
  }
}

// Tests
const assert = (n, c) => { if(c){console.log(`  ${n.padEnd(50)}PASS`);return 1;} console.log(`  ${n.padEnd(50)}FAIL`);return 0; };
const bc = (...v) => new Uint8Array(v);
let p=0;
console.log('\nFLUX JavaScript VM Tests\n==================\n');
{const v=new FluxVM();v.execute(bc(0x00));p+=assert('halt',v.halted&&v.cycles===1);}
{const v=new FluxVM();v.execute(bc(0x18,0,42,0x00));p+=assert('movi',v.gp[0]===42);}
{const v=new FluxVM();v.execute(bc(0x18,0,0xFB,0x00));p+=assert('movi_neg',v.gp[0]===-5);}
{const v=new FluxVM();v.execute(bc(0x40,0,0xE8,0x03,0x00));p+=assert('movi16',v.gp[0]===1000);}
{const v=new FluxVM();v.execute(bc(0x18,0,10,0x18,1,20,0x20,2,0,1,0x00));p+=assert('add',v.gp[2]===30);}
{const v=new FluxVM();v.execute(bc(0x18,0,7,0x18,1,6,0x22,2,0,1,0x00));p+=assert('mul',v.gp[2]===42);}
{const v=new FluxVM();v.execute(bc(0x18,0,42,0x18,1,7,0x23,2,0,1,0x00));p+=assert('div',v.gp[2]===6);}
{const v=new FluxVM();v.execute(bc(0x18,0,10,0x08,0,0x08,0,0x09,0,0x00));p+=assert('inc_dec',v.gp[0]===11);}
{const v=new FluxVM();v.execute(bc(0x18,0,42,0x0C,0,0x18,0,0,0x0D,1,0x00));p+=assert('push_pop',v.gp[1]===42);}
{const v=new FluxVM();v.execute(bc(0x18,0,5,0x18,1,0,0x08,1,0x09,0,0x3D,0,0xFC,0,0x00));p+=assert('jnz_loop',v.gp[0]===0&&v.gp[1]===5);}
{const v=new FluxVM();v.execute(bc(0x18,0,1,0x18,1,1,0x18,2,10,0x20,3,0,1,0x3A,0,1,0,0x3A,1,3,0,0x09,2,0x3D,2,0xF2,0,0x00));p+=assert('fibonacci',v.gp[1]===144);}
console.log(`\n==================\nResults: ${p} passed, ${11-p} failed\n`);
