#include "flux_vm.hpp"
#include <cstdio>

#define OP_HALT 0x00
#define OP_INC 0x08
#define OP_DEC 0x09
#define OP_PUSH 0x0C
#define OP_POP 0x0D
#define OP_MOVI 0x18
#define OP_ADD 0x20
#define OP_MUL 0x22
#define OP_CMP_GT 0x2E
#define OP_MOV 0x3A
#define OP_JNZ 0x3D

int main() {
    printf("\nDojo Level 1 - Validation\n========================\n\n");
    int p=0, f=0;
    
    // Ex1: First Light
    { FluxVM vm; vm.execute({OP_MOVI,0,42,OP_HALT});
      printf("  %s ex1_first_light (R0=%d)\n", vm.gp[0]==42?"PASS":"FAIL", vm.gp[0]);
      if(vm.gp[0]==42) p++; else f++; }

    // Ex2: Double Down
    { FluxVM vm; vm.execute({OP_MOVI,0,7,OP_MOVI,1,7,OP_ADD,2,0,1,OP_HALT});
      printf("  %s ex2_double_down (R2=%d)\n", vm.gp[2]==14?"PASS":"FAIL", vm.gp[2]);
      if(vm.gp[2]==14) p++; else f++; }

    // Ex3: Counter
    { FluxVM vm; vm.execute({OP_MOVI,0,5,OP_MOVI,1,0,OP_INC,1,OP_DEC,0,OP_JNZ,0,252,0,OP_HALT});
      bool ok = vm.gp[0]==0 && vm.gp[1]==5;
      printf("  %s ex3_counter (R0=%d,R1=%d)\n", ok?"PASS":"FAIL", vm.gp[0], vm.gp[1]);
      if(ok) p++; else f++; }

    // Ex4: Stack Play
    { FluxVM vm; vm.execute({OP_MOVI,0,10,OP_MOVI,1,20,OP_PUSH,0,OP_PUSH,1,OP_POP,0,OP_POP,1,OP_HALT});
      bool ok = vm.gp[0]==20 && vm.gp[1]==10;
      printf("  %s ex4_stack_play (R0=%d,R1=%d)\n", ok?"PASS":"FAIL", vm.gp[0], vm.gp[1]);
      if(ok) p++; else f++; }

    // Ex5: Factorial(6)
    { FluxVM vm; vm.execute({OP_MOVI,0,6,OP_MOVI,1,1,OP_MUL,1,1,0,OP_DEC,0,OP_JNZ,0,250,0,OP_HALT});
      printf("  %s ex5_factorial (R1=%d)\n", vm.gp[1]==720?"PASS":"FAIL", vm.gp[1]);
      if(vm.gp[1]==720) p++; else f++; }

    // Ex6: Comparison
    { FluxVM vm; vm.execute({OP_MOVI,0,15,OP_MOVI,1,25,OP_CMP_GT,2,0,1,OP_CMP_GT,3,1,0,OP_HALT});
      bool ok = vm.gp[2]==0 && vm.gp[3]==1;
      printf("  %s ex6_comparison (R2=%d,R3=%d)\n", ok?"PASS":"FAIL", vm.gp[2], vm.gp[3]);
      if(ok) p++; else f++; }

    // Ex7: Fibonacci
    { FluxVM vm; vm.execute({OP_MOVI,0,1,OP_MOVI,1,1,OP_MOVI,2,10,OP_ADD,3,0,1,OP_MOV,0,1,0,OP_MOV,1,3,0,OP_DEC,2,OP_JNZ,2,242,0,OP_HALT});
      printf("  %s ex7_fibonacci (R1=%d)\n", vm.gp[1]==144?"PASS":"FAIL", vm.gp[1]);
      if(vm.gp[1]==144) p++; else f++; }

    printf("\n========================\nResults: %d/7 passed\n", p);
    if(f==0) printf("\nLevel 1 Complete: Hello Fleet!\n\n");
    return f;
}
