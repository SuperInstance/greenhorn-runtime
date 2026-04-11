//! FLUX Unified ISA — Rust implementation
//! Matches Go/C/C++/Zig runtimes exactly.

const OP_HALT: u8 = 0x00;
const OP_NOP: u8 = 0x01;
const OP_INC: u8 = 0x08;
const OP_DEC: u8 = 0x09;
const OP_NOT: u8 = 0x0A;
const OP_NEG: u8 = 0x0B;
const OP_PUSH: u8 = 0x0C;
const OP_POP: u8 = 0x0D;
const OP_STRIPCONF: u8 = 0x17;
const OP_MOVI: u8 = 0x18;
const OP_ADDI: u8 = 0x19;
const OP_SUBI: u8 = 0x1A;
const OP_ADD: u8 = 0x20;
const OP_SUB: u8 = 0x21;
const OP_MUL: u8 = 0x22;
const OP_DIV: u8 = 0x23;
const OP_MOD: u8 = 0x24;
const OP_AND: u8 = 0x25;
const OP_OR: u8 = 0x26;
const OP_XOR: u8 = 0x27;
const OP_MOV: u8 = 0x3A;
const OP_JZ: u8 = 0x3C;
const OP_JNZ: u8 = 0x3D;
const OP_MOVI16: u8 = 0x40;
const OP_JMP: u8 = 0x43;
const OP_LOOP: u8 = 0x46;

const NUM_REGS: usize = 64;
const STACK_SIZE: usize = 4096;

struct FluxVM {
    gp: [i32; NUM_REGS],
    stack: [i32; STACK_SIZE],
    sp: usize,
    pc: i32,
    halted: bool,
    cycles: i32,
    strip_n: i32,
}

impl FluxVM {
    fn new() -> Self {
        FluxVM {
            gp: [0i32; NUM_REGS],
            stack: [0i32; STACK_SIZE],
            sp: STACK_SIZE,
            pc: 0,
            halted: false,
            cycles: 0,
            strip_n: 0,
        }
    }

    fn execute(&mut self, bc: &[u8]) -> i32 {
        while !self.halted && self.pc >= 0 && (self.pc as usize) < bc.len() {
            let result = self.step(bc);
            if result < 0 { break; }
            if result > 0 { self.pc += result; }
        }
        self.cycles
    }

    fn format_size(op: u8) -> i32 {
        match op {
            0..=0x07 => 1,
            0..=0x0F => 2,
            0..=0x17 => 2,
            0..=0x1F => 3,
            0..=0x4F => 4,
            _ => 1,
        }
    }

    fn step(&mut self, bc: &[u8]) -> i32 {
        let pc = self.pc as usize;
        if pc >= bc.len() { return -1; }
        let op = bc[pc];
        let sz = Self::format_size(op);
        if pc + (sz as usize) > bc.len() { self.halted = true; return -1; }
        self.cycles += 1;
        if self.strip_n > 0 { self.strip_n -= 1; }

        match op {
            OP_HALT => { self.halted = true; 1 },
            OP_NOP => 1,
            OP_INC => { let rd = bc[pc+1] as usize; self.gp[rd] += 1; 2 },
            OP_DEC => { let rd = bc[pc+1] as usize; self.gp[rd] -= 1; 2 },
            OP_NOT => { let rd = bc[pc+1] as usize; self.gp[rd] = !self.gp[rd]; 2 },
            OP_NEG => { let rd = bc[pc+1] as usize; self.gp[rd] = -self.gp[rd]; 2 },
            OP_PUSH => { self.sp -= 1; self.stack[self.sp] = self.gp[bc[pc+1] as usize]; 2 },
            OP_POP => { let rd = bc[pc+1] as usize; self.gp[rd] = self.stack[self.sp]; self.sp += 1; 2 },
            OP_STRIPCONF => { self.strip_n = bc[pc+1] as i32; 2 },
            OP_MOVI => { let rd = bc[pc+1] as usize; self.gp[rd] = bc[pc+2] as i8 as i32; 3 },
            OP_ADDI => { let rd = bc[pc+1] as usize; self.gp[rd] += bc[pc+2] as i8 as i32; 3 },
            OP_SUBI => { let rd = bc[pc+1] as usize; self.gp[rd] -= bc[pc+2] as i8 as i32; 3 },
            OP_ADD => { let rd = bc[pc+1] as usize; let r1 = bc[pc+2] as usize; let r2 = bc[pc+3] as usize;
                        self.gp[rd] = self.gp[r1].wrapping_add(self.gp[r2]); 4 },
            OP_SUB => { let rd = bc[pc+1] as usize; let r1 = bc[pc+2] as usize; let r2 = bc[pc+3] as usize;
                        self.gp[rd] = self.gp[r1].wrapping_sub(self.gp[r2]); 4 },
            OP_MUL => { let rd = bc[pc+1] as usize; let r1 = bc[pc+2] as usize; let r2 = bc[pc+3] as usize;
                        self.gp[rd] = self.gp[r1].wrapping_mul(self.gp[r2]); 4 },
            OP_DIV => { let rd = bc[pc+1] as usize; let r1 = bc[pc+2] as usize; let r2 = bc[pc+3] as usize;
                        if self.gp[r2] == 0 { self.halted = true; return 4; }
                        self.gp[rd] = self.gp[r1] / self.gp[r2]; 4 },
            OP_MOD => { let rd = bc[pc+1] as usize; let r1 = bc[pc+2] as usize; let r2 = bc[pc+3] as usize;
                        self.gp[rd] = self.gp[r1] % self.gp[r2]; 4 },
            OP_AND => { let rd = bc[pc+1] as usize; let r1 = bc[pc+2] as usize; let r2 = bc[pc+3] as usize;
                        self.gp[rd] = self.gp[r1] & self.gp[r2]; 4 },
            OP_OR  => { let rd = bc[pc+1] as usize; let r1 = bc[pc+2] as usize; let r2 = bc[pc+3] as usize;
                        self.gp[rd] = self.gp[r1] | self.gp[r2]; 4 },
            OP_XOR => { let rd = bc[pc+1] as usize; let r1 = bc[pc+2] as usize; let r2 = bc[pc+3] as usize;
                        self.gp[rd] = self.gp[r1] ^ self.gp[r2]; 4 },
            OP_MOV => { let rd = bc[pc+1] as usize; let r1 = bc[pc+2] as usize;
                        self.gp[rd] = self.gp[r1]; 4 },
            OP_JNZ => { let rd = bc[pc+1] as usize; let off = bc[pc+2] as i8 as i32;
                        if self.gp[rd] != 0 { self.pc += off; return 0; } 4 },
            OP_JZ  => { let rd = bc[pc+1] as usize; let off = bc[pc+2] as i8 as i32;
                        if self.gp[rd] == 0 { self.pc += off; return 0; } 4 },
            OP_MOVI16 => { let rd = bc[pc+1] as usize;
                           let imm = (bc[pc+2] as u16) | ((bc[pc+3] as u16) << 8);
                           self.gp[rd] = imm as i16 as i32; 4 },
            OP_JMP => { let off = ((bc[pc+2] as u16) | ((bc[pc+3] as u16) << 8)) as i16 as i32;
                        self.pc += off; 0 },
            OP_LOOP => { let rd = bc[pc+1] as usize; let off = (bc[pc+2] as u16) | ((bc[pc+3] as u16) << 8);
                         self.gp[rd] -= 1;
                         if self.gp[rd] > 0 { self.pc -= off as i32; return 0; } 4 },
            _ => sz,
        }
    }
}

// ── Tests ──────────────────────────────────────────────

macro_rules! assert_eq {
    ($a:expr, $b:expr) => {
        if $a != $b { return Err(format!("{} != {} (line {})", stringify!($a), $b, line!())); }
    };
}

type TResult = Result<(), String>;

fn test_halt() -> TResult {
    let mut vm = FluxVM::new();
    vm.execute(&[OP_HALT]);
    assert_eq!(vm.halted, true);
    assert_eq!(vm.cycles, 1);
    Ok(())
}

fn test_movi() -> TResult {
    let mut vm = FluxVM::new();
    vm.execute(&[OP_MOVI, 0, 42, OP_HALT]);
    assert_eq!(vm.gp[0], 42);
    Ok(())
}

fn test_movi_neg() -> TResult {
    let mut vm = FluxVM::new();
    vm.execute(&[OP_MOVI, 0, 0xFB, OP_HALT]); // -5
    assert_eq!(vm.gp[0], -5);
    Ok(())
}

fn test_movi16() -> TResult {
    let mut vm = FluxVM::new();
    vm.execute(&[OP_MOVI16, 0, 0xE8, 0x03, OP_HALT]);
    assert_eq!(vm.gp[0], 1000);
    Ok(())
}

fn test_movi16_neg() -> TResult {
    let mut vm = FluxVM::new();
    vm.execute(&[OP_MOVI16, 0, 0x00, 0x80, OP_HALT]);
    assert_eq!(vm.gp[0], -32768);
    Ok(())
}

fn test_add() -> TResult {
    let mut vm = FluxVM::new();
    vm.execute(&[OP_MOVI,0,10, OP_MOVI,1,20, OP_ADD,2,0,1, OP_HALT]);
    assert_eq!(vm.gp[2], 30);
    Ok(())
}

fn test_sub() -> TResult {
    let mut vm = FluxVM::new();
    vm.execute(&[OP_MOVI,0,50, OP_MOVI,1,20, OP_SUB,2,0,1, OP_HALT]);
    assert_eq!(vm.gp[2], 30);
    Ok(())
}

fn test_mul() -> TResult {
    let mut vm = FluxVM::new();
    vm.execute(&[OP_MOVI,0,7, OP_MOVI,1,6, OP_MUL,2,0,1, OP_HALT]);
    assert_eq!(vm.gp[2], 42);
    Ok(())
}

fn test_div() -> TResult {
    let mut vm = FluxVM::new();
    vm.execute(&[OP_MOVI,0,42, OP_MOVI,1,7, OP_DIV,2,0,1, OP_HALT]);
    assert_eq!(vm.gp[2], 6);
    Ok(())
}

fn test_inc_dec() -> TResult {
    let mut vm = FluxVM::new();
    vm.execute(&[OP_MOVI,0,10, OP_INC,0, OP_INC,0, OP_DEC,0, OP_HALT]);
    assert_eq!(vm.gp[0], 11);
    Ok(())
}

fn test_push_pop() -> TResult {
    let mut vm = FluxVM::new();
    vm.execute(&[OP_MOVI,0,42, OP_PUSH,0, OP_MOVI,0,0, OP_POP,1, OP_HALT]);
    assert_eq!(vm.gp[1], 42);
    Ok(())
}

fn test_jnz_loop() -> TResult {
    let mut vm = FluxVM::new();
    vm.execute(&[OP_MOVI,0,5, OP_MOVI,1,0, OP_INC,1, OP_DEC,0, OP_JNZ,0,252,0, OP_HALT]); // -4 = 252u8
    assert_eq!(vm.gp[0], 0);
    assert_eq!(vm.gp[1], 5);
    Ok(())
}

fn test_fibonacci() -> TResult {
    let mut vm = FluxVM::new();
    vm.execute(&[
        OP_MOVI,0,1, OP_MOVI,1,1, OP_MOVI,2,10,
        OP_ADD,3,0,1, OP_MOV,0,1,0, OP_MOV,1,3,0, OP_DEC,2,
        OP_JNZ,2,242,0, // -14 = 242u8
        OP_HALT
    ]);
    assert_eq!(vm.gp[1], 144);
    Ok(())
}

fn main() {
    let tests: Vec<(&str, fn() -> TResult)> = vec![
        ("halt", test_halt),
        ("movi", test_movi),
        ("movi_neg", test_movi_neg),
        ("movi16", test_movi16),
        ("movi16_neg", test_movi16_neg),
        ("add", test_add),
        ("sub", test_sub),
        ("mul", test_mul),
        ("div", test_div),
        ("inc_dec", test_inc_dec),
        ("push_pop", test_push_pop),
        ("jnz_loop", test_jnz_loop),
        ("fibonacci", test_fibonacci),
    ];

    println!("\nFLUX Rust VM Tests\n==================\n");
    let mut passed = 0usize;
    let mut failed = 0usize;
    for (name, f) in &tests {
        print!("  {:<50}", name);
        match f() {
            Ok(()) => { println!("PASS"); passed += 1; }
            Err(e) => { println!("FAIL: {}", e); failed += 1; }
        }
    }
    println!("\n==================\nResults: {} passed, {} failed", passed, failed);
    if failed > 0 { std::process::exit(1); }
}
