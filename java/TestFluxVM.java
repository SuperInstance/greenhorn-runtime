public class TestFluxVM {
    static int passed = 0, failed = 0;
    
    static void check(String name, boolean cond) {
        if (cond) { System.out.printf("  %-50sPASS%n", name); passed++; }
        else { System.out.printf("  %-50sFAIL%n", name); failed++; }
    }
    
    static byte[] bc(int... vals) {
        byte[] r = new byte[vals.length];
        for (int i = 0; i < vals.length; i++) r[i] = (byte)vals[i];
        return r;
    }

    public static void main(String[] args) {
        System.out.println("\nFLUX Java VM Tests\n==================\n");

        { FluxVM v = new FluxVM(); v.execute(bc(0x00)); check("halt", v.halted && v.cycles==1); }
        { FluxVM v = new FluxVM(); v.execute(bc(0x01,0x00)); check("nop", v.cycles==2); }
        { FluxVM v = new FluxVM(); v.execute(bc(0x18,0,42,0x00)); check("movi", v.gp[0]==42); }
        { FluxVM v = new FluxVM(); v.execute(bc(0x18,0,0xFB,0x00)); check("movi_neg", v.gp[0]==-5); }
        { FluxVM v = new FluxVM(); v.execute(bc(0x40,0,0xE8,0x03,0x00)); check("movi16", v.gp[0]==1000); }
        { FluxVM v = new FluxVM(); v.execute(bc(0x40,0,0x00,0x80,0x00)); check("movi16_neg", v.gp[0]==-32768); }
        { FluxVM v = new FluxVM(); v.execute(bc(0x18,0,10,0x18,1,20,0x20,2,0,1,0x00)); check("add", v.gp[2]==30); }
        { FluxVM v = new FluxVM(); v.execute(bc(0x18,0,7,0x18,1,6,0x22,2,0,1,0x00)); check("mul", v.gp[2]==42); }
        { FluxVM v = new FluxVM(); v.execute(bc(0x18,0,42,0x18,1,7,0x23,2,0,1,0x00)); check("div", v.gp[2]==6); }
        { FluxVM v = new FluxVM(); v.execute(bc(0x18,0,10,0x08,0,0x08,0,0x09,0,0x00)); check("inc_dec", v.gp[0]==11); }
        { FluxVM v = new FluxVM(); v.execute(bc(0x18,0,42,0x0C,0,0x18,0,0,0x0D,1,0x00)); check("push_pop", v.gp[1]==42); }
        { FluxVM v = new FluxVM(); v.execute(bc(0x18,0,3,0x18,1,7,0x2A,2,0,1,0x2B,3,0,1,0x00)); check("min_max", v.gp[2]==3 && v.gp[3]==7); }
        { FluxVM v = new FluxVM(); v.execute(bc(0x18,0,5,0x18,1,0,0x08,1,0x09,0,0x3D,0,0xFC,0,0x00)); check("jnz_loop", v.gp[0]==0 && v.gp[1]==5); }
        { FluxVM v = new FluxVM(); // fibonacci
            v.execute(bc(0x18,0,1,0x18,1,1,0x18,2,10, 0x20,3,0,1,0x3A,0,1,0,0x3A,1,3,0,0x09,2, 0x3D,2,0xF2,0, 0x00));
            check("fibonacci", v.gp[1]==144);
        }

        System.out.printf("%n==================%nResults: %d passed, %d failed%n", passed, failed);
        if (failed > 0) System.exit(1);
    }
}
