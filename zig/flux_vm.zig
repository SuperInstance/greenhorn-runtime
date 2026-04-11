const std = @import("std");

const Op = struct {
    pub const HALT: u8 = 0x00;
    pub const NOP: u8 = 0x01;
    pub const INC: u8 = 0x08;
    pub const DEC: u8 = 0x09;
    pub const NOT: u8 = 0x0A;
    pub const NEG: u8 = 0x0B;
    pub const PUSH: u8 = 0x0C;
    pub const POP: u8 = 0x0D;
    pub const STRIPCONF: u8 = 0x17;
    pub const MOVI: u8 = 0x18;
    pub const ADDI: u8 = 0x19;
    pub const SUBI: u8 = 0x1A;
    pub const ADD: u8 = 0x20;
    pub const SUB: u8 = 0x21;
    pub const MUL: u8 = 0x22;
    pub const DIV: u8 = 0x23;
    pub const MOD: u8 = 0x24;
    pub const AND: u8 = 0x25;
    pub const OR: u8 = 0x26;
    pub const XOR: u8 = 0x27;
    pub const MOV: u8 = 0x3A;
    pub const JNZ: u8 = 0x3D;
    pub const JZ: u8 = 0x3C;
    pub const MOVI16: u8 = 0x40;
    pub const JMP: u8 = 0x43;
    pub const LOOP: u8 = 0x46;
};

pub const NUM_REGS = 64;
pub const STACK_SIZE = 4096;

pub const FluxVM = struct {
    gp: [NUM_REGS]i32,
    stack: [STACK_SIZE]i32,
    sp: i32,
    pc: i32,
    halted: bool,
    cycles: i32,
    strip_n: i32,

    pub fn init() FluxVM {
        return FluxVM{
            .gp = [_]i32{0} ** NUM_REGS,
            .stack = [_]i32{0} ** STACK_SIZE,
            .sp = STACK_SIZE,
            .pc = 0,
            .halted = false,
            .cycles = 0,
            .strip_n = 0,
        };
    }

    pub fn execute(self: *FluxVM, bc: []const u8) i32 {
        while (!self.halted and self.pc < @as(i32, @intCast(bc.len))) {
            const result = self.step(bc);
            if (result < 0) break;
            if (result > 0) self.pc += result;
        }
        return self.cycles;
    }

    fn step(self: *FluxVM, bc: []const u8) i32 {
        if (self.pc >= @as(i32, @intCast(bc.len))) return -1;
        const op = bc[@as(usize, @intCast(self.pc))];
        const sz = formatSize(op);
        if (self.pc + sz > @as(i32, @intCast(bc.len))) { self.halted = true; return -1; }
        self.cycles += 1;
        if (self.strip_n > 0) self.strip_n -= 1;

        switch (op) {
            Op.HALT => { self.halted = true; return 1; },
            Op.NOP => return 1,
            Op.INC => { self.gp[bc[@intCast(self.pc+1)]] += 1; return 2; },
            Op.DEC => { self.gp[bc[@intCast(self.pc+1)]] -= 1; return 2; },
            Op.NOT => { const rd = bc[@intCast(self.pc+1)]; self.gp[rd] = ~self.gp[rd]; return 2; },
            Op.NEG => { const rd = bc[@intCast(self.pc+1)]; self.gp[rd] = -self.gp[rd]; return 2; },
            Op.PUSH => { self.sp -= 1; self.stack[@intCast(self.sp)] = self.gp[bc[@intCast(self.pc+1)]]; return 2; },
            Op.POP => { const rd = bc[@intCast(self.pc+1)]; self.gp[rd] = self.stack[@intCast(self.sp)]; self.sp += 1; return 2; },
            Op.STRIPCONF => { self.strip_n = @intCast(bc[@intCast(self.pc+1)]); return 2; },
            Op.MOVI => { const rd = bc[@intCast(self.pc+1)]; const imm: i8 = @bitCast(bc[@intCast(self.pc+2)]); self.gp[rd] = @intCast(imm); return 3; },
            Op.ADDI => { const rd = bc[@intCast(self.pc+1)]; const imm: i8 = @bitCast(bc[@intCast(self.pc+2)]); self.gp[@intCast(rd)] += @intCast(imm); return 3; },
            Op.SUBI => { const rd = bc[@intCast(self.pc+1)]; const imm: i8 = @bitCast(bc[@intCast(self.pc+2)]); self.gp[@intCast(rd)] -= @intCast(imm); return 3; },
            Op.ADD => { self.gp[bc[@intCast(self.pc+1)]] = self.gp[bc[@intCast(self.pc+2)]] + self.gp[bc[@intCast(self.pc+3)]]; return 4; },
            Op.SUB => { self.gp[bc[@intCast(self.pc+1)]] = self.gp[bc[@intCast(self.pc+2)]] - self.gp[bc[@intCast(self.pc+3)]]; return 4; },
            Op.MUL => { self.gp[bc[@intCast(self.pc+1)]] = self.gp[bc[@intCast(self.pc+2)]] * self.gp[bc[@intCast(self.pc+3)]]; return 4; },
            Op.DIV => {
                if (self.gp[bc[@intCast(self.pc+3)]] == 0) { self.halted = true; return 4; }
                self.gp[bc[@intCast(self.pc+1)]] = @divTrunc(self.gp[bc[@intCast(self.pc+2)]], self.gp[bc[@intCast(self.pc+3)]]);
                return 4;
            },
            Op.MOD => { self.gp[bc[@intCast(self.pc+1)]] = @mod(self.gp[bc[@intCast(self.pc+2)]], self.gp[bc[@intCast(self.pc+3)]]); return 4; },
            Op.AND => { self.gp[bc[@intCast(self.pc+1)]] = self.gp[bc[@intCast(self.pc+2)]] & self.gp[bc[@intCast(self.pc+3)]]; return 4; },
            Op.OR  => { self.gp[bc[@intCast(self.pc+1)]] = self.gp[bc[@intCast(self.pc+2)]] | self.gp[bc[@intCast(self.pc+3)]]; return 4; },
            Op.MOV => { self.gp[bc[@intCast(self.pc+1)]] = self.gp[bc[@intCast(self.pc+2)]]; return 4; },
            Op.JNZ => {
                const rd = bc[@intCast(self.pc+1)];
                const off: i32 = @intCast(@as(i8, @bitCast(bc[@intCast(self.pc+2)])));
                if (self.gp[rd] != 0) { self.pc += off; return 0; }
                return 4;
            },
            Op.JZ => {
                const rd = bc[@intCast(self.pc+1)];
                const off: i32 = @intCast(@as(i8, @bitCast(bc[@intCast(self.pc+2)])));
                if (self.gp[rd] == 0) { self.pc += off; return 0; }
                return 4;
            },
            Op.MOVI16 => {
                const rd = bc[@intCast(self.pc+1)];
                const imm: i16 = @bitCast(@as(u16, bc[@intCast(self.pc+2)]) | (@as(u16, bc[@intCast(self.pc+3)]) << 8));
                self.gp[rd] = @intCast(imm);
                return 4;
            },
            Op.LOOP => {
                const rd = bc[@intCast(self.pc+1)];
                const off: u16 = @as(u16, bc[@intCast(self.pc+2)]) | (@as(u16, bc[@intCast(self.pc+3)]) << 8);
                self.gp[rd] -= 1;
                if (self.gp[rd] > 0) { self.pc -= @intCast(off); return 0; }
                return 4;
            },
            else => return sz,
        }
    }

    fn formatSize(op: u8) i32 {
        return if (op <= 0x07) 1
            else if (op <= 0x0F) 2
            else if (op <= 0x17) 2
            else if (op <= 0x1F) 3
            else if (op <= 0x4F) 4
            else 1;
    }
};

test "halt" {
    var vm = FluxVM.init();
    const bc = [_]u8{Op.HALT};
    _ = vm.execute(&bc);
    try std.testing.expect(vm.halted);
    try std.testing.expect(vm.cycles == 1);
}

test "movi" {
    var vm = FluxVM.init();
    const bc = [_]u8{ Op.MOVI, 0, 42, Op.HALT };
    _ = vm.execute(&bc);
    try std.testing.expect(vm.gp[0] == 42);
}

test "add" {
    var vm = FluxVM.init();
    const bc = [_]u8{ Op.MOVI, 0, 10, Op.MOVI, 1, 20, Op.ADD, 2, 0, 1, Op.HALT };
    _ = vm.execute(&bc);
    try std.testing.expect(vm.gp[2] == 30);
}

test "mul" {
    var vm = FluxVM.init();
    const bc = [_]u8{ Op.MOVI, 0, 7, Op.MOVI, 1, 6, Op.MUL, 2, 0, 1, Op.HALT };
    _ = vm.execute(&bc);
    try std.testing.expect(vm.gp[2] == 42);
}

test "inc_dec" {
    var vm = FluxVM.init();
    const bc = [_]u8{ Op.MOVI, 0, 10, Op.INC, 0, Op.INC, 0, Op.DEC, 0, Op.HALT };
    _ = vm.execute(&bc);
    try std.testing.expect(vm.gp[0] == 11);
}

test "push_pop" {
    var vm = FluxVM.init();
    const bc = [_]u8{ Op.MOVI, 0, 42, Op.PUSH, 0, Op.MOVI, 0, 0, Op.POP, 1, Op.HALT };
    _ = vm.execute(&bc);
    try std.testing.expect(vm.gp[1] == 42);
}

test "movi16" {
    var vm = FluxVM.init();
    const bc = [_]u8{ Op.MOVI16, 0, 0xE8, 0x03, Op.HALT };
    _ = vm.execute(&bc);
    try std.testing.expect(vm.gp[0] == 1000);
}

test "fibonacci" {
    var vm = FluxVM.init();
    const bc = [_]u8{
        Op.MOVI, 0, 1,  Op.MOVI, 1, 1,  Op.MOVI, 2, 10,
        Op.ADD, 3, 0, 1,  Op.MOV, 0, 1, 0,  Op.MOV, 1, 3, 0,
        Op.DEC, 2,  Op.JNZ, 2, @bitCast(@as(i8, -14)), 0,
        Op.HALT
    };
    _ = vm.execute(&bc);
    try std.testing.expect(vm.gp[1] == 144);
}

const expect = std.testing.expect;
