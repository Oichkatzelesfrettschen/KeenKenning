#ifndef KEEN_INTERNAL_H
#define KEEN_INTERNAL_H

#include <inttypes.h>

#include "keen_modes.h"
#include "latin.h"
#include "puzzles.h"

/*
 * Difficulty levels: X-macro pattern keeps enums and naming in sync.
 */
#define DIFFLIST(A)                                       \
    A(EASY, Easy, solver_easy, e)                         \
    A(NORMAL, Normal, solver_normal, n)                   \
    A(HARD, Hard, solver_hard, h)                         \
    A(EXTREME, Extreme, solver_extreme, x)                \
    A(UNREASONABLE, Unreasonable, solver_unreasonable, u) \
    A(LUDICROUS, Ludicrous, solver_ludicrous, l)          \
    A(INCOMPREHENSIBLE, Incomprehensible, solver_incomprehensible, i)
#define ENUM(upper, title, func, lower) DIFF_##upper,
enum { DIFFLIST(ENUM) DIFFCOUNT };
#undef ENUM

/*
 * Clue notation: Operation codes stored in high bits of clue value.
 */
#define C_ADD 0x00000000UL
#define C_MUL 0x20000000UL
#define C_SUB 0x40000000UL
#define C_DIV 0x60000000UL
#define C_EXP UINT64_C(0x80000000)
#define C_MOD UINT64_C(0xA0000000)
#define C_GCD UINT64_C(0xC0000000)
#define C_LCM UINT64_C(0xE0000000)
#define C_XOR UINT64_C(0x10000000)
#define C_AND UINT64_C(0x30000000)
#define C_OR  UINT64_C(0x50000000)
#define CMASK UINT64_C(0xF0000000)
#define CUNIT UINT64_C(0x10000000)

#endif
