#include "flux_vm.hpp"
#include <cassert>
#include <cstdio>
#include <vector>

static int passed = 0, failed = 0;
#define TEST(name) printf("  %-50s", #name);
#define PASS() do { printf("PASS\n"); passed++; } while(0)
#define ASSERT_EQ(a, b) do { if ((a) != (b)) { printf("FAIL: %s=%lld, expected %lld\n", #a, (long long)(a), (long long)(b)); failed++; return; } } while(0)

void test_halt() { TEST(halt); FluxVM vm; vm.execute({OP_HALT}); assert(vm.halted); assert(vm.cycles==1); PASS(); }
void test_nop() { TEST(nop); FluxVM vm; vm.execute({OP_NOP, OP_HALT}); assert(vm.cycles==2); PASS(); }
void test_movi() { TEST(movi); FluxVM vm; vm.execute({OP_MOVI, 0, 42, OP_HALT}); assert(vm.gp[0]==42); PASS(); }
void test_movi_neg() { TEST(movi_neg); FluxVM vm; vm.execute({OP_MOVI, 0, (uint8_t)-5, OP_HALT}); assert(vm.gp[0]==-5); PASS(); }
void test_movi16() { TEST(movi16); FluxVM vm; vm.execute({OP_MOVI16, 0, 0xE8, 0x03, OP_HALT}); assert(vm.gp[0]==1000); PASS(); }
void test_movi16_neg() { TEST(movi16_neg); FluxVM vm; vm.execute({OP_MOVI16, 0, 0x00, 0x80, OP_HALT}); assert(vm.gp[0]==-32768); PASS(); }
void test_add() { TEST(add); FluxVM vm; vm.execute({OP_MOVI,0,10, OP_MOVI,1,20, OP_ADD,2,0,1, OP_HALT}); assert(vm.gp[2]==30); PASS(); }
void test_mul() { TEST(mul); FluxVM vm; vm.execute({OP_MOVI,0,7, OP_MOVI,1,6, OP_MUL,2,0,1, OP_HALT}); assert(vm.gp[2]==42); PASS(); }
void test_div() { TEST(div); FluxVM vm; vm.execute({OP_MOVI,0,42, OP_MOVI,1,7, OP_DIV,2,0,1, OP_HALT}); assert(vm.gp[2]==6); PASS(); }
void test_inc_dec() { TEST(inc_dec); FluxVM vm; vm.execute({OP_MOVI,0,10, OP_INC,0, OP_INC,0, OP_DEC,0, OP_HALT}); assert(vm.gp[0]==11); PASS(); }
void test_push_pop() { TEST(push_pop); FluxVM vm; vm.execute({OP_MOVI,0,42, OP_PUSH,0, OP_MOVI,0,0, OP_POP,1, OP_HALT}); assert(vm.gp[1]==42); PASS(); }
void test_cmp_eq() { TEST(cmp_eq); FluxVM vm; vm.execute({OP_MOVI,0,5, OP_MOVI,1,5, OP_CMP_EQ,2,0,1, OP_HALT}); assert(vm.gp[2]==1); PASS(); }
void test_min_max() { TEST(min_max); FluxVM vm; vm.execute({OP_MOVI,0,3, OP_MOVI,1,7, OP_MIN,2,0,1, OP_MAX,3,0,1, OP_HALT}); assert(vm.gp[2]==3); assert(vm.gp[3]==7); PASS(); }
void test_jnz_loop() {
    TEST(jnz_loop);
    FluxVM vm;
    vm.execute({OP_MOVI,0,5, OP_MOVI,1,0, OP_INC,1, OP_DEC,0, OP_JNZ,0,(uint8_t)-4,0, OP_HALT});
    assert(vm.gp[0]==0); assert(vm.gp[1]==5); PASS();
}
void test_fibonacci() {
    TEST(fibonacci);
    FluxVM vm;
    vm.execute({
        OP_MOVI,0,1, OP_MOVI,1,1, OP_MOVI,2,10,
        OP_ADD,3,0,1, OP_MOV,0,1,0, OP_MOV,1,3,0, OP_DEC,2,
        OP_JNZ,2,(uint8_t)-14,0, OP_HALT
    });
    assert(vm.gp[1]==144); PASS();
}

int main() {
    printf("\nFLUX C++ VM Tests\n==================\n\n");
    test_halt(); test_nop(); test_movi(); test_movi_neg(); test_movi16();
    test_movi16_neg(); test_add(); test_mul(); test_div(); test_inc_dec();
    test_push_pop(); test_cmp_eq(); test_min_max(); test_jnz_loop(); test_fibonacci();
    printf("\n==================\nResults: %d passed, %d failed\n", passed, failed);
    return failed;
}
