/*
 * kenken.c: Core KenKen puzzle generation and solving engine
 *
 * KenKen is a Latin square puzzle where players fill an NxN grid such that
 * each row and column contains the digits 1-N exactly once. The grid is
 * divided into "cages" (irregular regions) with arithmetic clues that
 * constrain the digits within each cage.
 *
 * This implementation supports:
 *   - Standard addition, subtraction, multiplication, and division cages
 *   - Extended modes: exponentiation, zero-inclusive, negative numbers
 *   - Killer Sudoku constraint (no repeated digits in cages)
 *   - Multiple difficulty levels with distinct solving strategies
 *
 * Originally derived from Simon Tatham's Portable Puzzle Collection,
 * substantially extended for Orthogon with additional game modes,
 * AI-assisted generation, and Android JNI integration.
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2004-2024 Simon Tatham
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 *
 * Based on work from Simon Tatham's Portable Puzzle Collection
 * Original source: https://www.chiark.greenend.org.uk/~sgtatham/puzzles/
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kenken.h"
#include "latin.h"
#include "puzzles.h"

/*
 * Difficulty levels: X-macro pattern ensures the enum values and
 * their string representations stay synchronized automatically.
 */
#define DIFFLIST(A)                                                                                \
    A(EASY, Easy, solver_easy, e)                                                                  \
    A(NORMAL, Normal, solver_normal, n)                                                            \
    A(HARD, Hard, solver_hard, h)                                                                  \
    A(EXTREME, Extreme, NULL, x)                                                                   \
    A(UNREASONABLE, Unreasonable, NULL, u)
#define ENUM(upper, title, func, lower) DIFF_##upper,
#define TITLE(upper, title, func, lower) #title,
#define ENCODE(upper, title, func, lower) #lower
#define CONFIG(upper, title, func, lower) ":" #title
enum { DIFFLIST(ENUM) DIFFCOUNT };
static char const *const keen_diffnames[] = {DIFFLIST(TITLE)};
static char const keen_diffchars[] = DIFFLIST(ENCODE);
#define DIFFCONFIG DIFFLIST(CONFIG)

/*
 * Clue notation: Operation codes stored in high bits of clue value.
 * Ordering matters: ADD/MUL before SUB/DIV ensures preference during
 * generation when multiple operations yield the same result.
 *
 * 8 operations using 3 bits (0-7):
 *   0: ADD  - Sum of all digits
 *   1: MUL  - Product of all digits
 *   2: SUB  - Absolute difference (2-cell only)
 *   3: DIV  - Integer quotient (2-cell only)
 *   4: EXP  - Exponentiation base^exp (2-cell, requires MODE_EXPONENT)
 *   5: MOD  - Modulo operation (2-cell, requires MODE_NUMBER_THEORY)
 *   6: GCD  - Greatest common divisor (requires MODE_NUMBER_THEORY)
 *   7: LCM  - Least common multiple (requires MODE_NUMBER_THEORY)
 */
#define C_ADD 0x00000000L
#define C_MUL 0x20000000L
#define C_SUB 0x40000000L
#define C_DIV 0x60000000L
#define C_EXP 0x80000000L  /* Exponentiation: a^b (requires MODE_EXPONENT) */
#define C_MOD 0xA0000000L  /* Modulo: a % b = clue (requires MODE_NUMBER_THEORY) */
#define C_GCD 0xC0000000L  /* GCD of all digits (requires MODE_NUMBER_THEORY) */
#define C_LCM 0xE0000000L  /* LCM of all digits (requires MODE_NUMBER_THEORY) */
#define CMASK 0xE0000000L  /* 3 bits for 8 operations (all slots used) */
#define CUNIT 0x20000000L

/*
 * Helper functions for number theory operations.
 */
static long gcd_helper(long a, long b) {
    while (b != 0) {
        long t = b;
        b = a % b;
        a = t;
    }
    return a < 0 ? -a : a;
}

static long lcm_helper(long a, long b) {
    if (a == 0 || b == 0) return 0;
    long g = gcd_helper(a, b);
    return (a / g) * b;  /* Avoid overflow by dividing first */
}

static long gcd_array(digit *arr, int n) {
    if (n == 0) return 0;
    long result = arr[0];
    for (int i = 1; i < n; i++) {
        result = gcd_helper(result, arr[i]);
        if (result == 1) return 1;  /* Early exit: GCD can't go lower */
    }
    return result;
}

static long lcm_array(digit *arr, int n) {
    if (n == 0) return 0;
    long result = arr[0];
    for (int i = 1; i < n; i++) {
        result = lcm_helper(result, arr[i]);
        if (result > 1000000) return result;  /* Prevent overflow */
    }
    return result;
}

/*
 * Maximum cage size. Larger cages increase solver complexity exponentially
 * (multiplicative clues with 7+ cells can exceed 32-bit value limits)
 * and create UI challenges (clue text overflow).
 *
 * MAXBLK_STANDARD: Default for classic KenKen (6 cells max)
 * MAXBLK_KILLER: Extended for Killer Sudoku mode (16 cells max)
 *
 * The clues array uses 'long' (64-bit on most platforms) to handle
 * products of large cages: e.g., 9^16 = 1.85e15 fits in 64 bits.
 */
#define MAXBLK_STANDARD 6
#define MAXBLK_KILLER   16
#define MAXBLK          MAXBLK_STANDARD  /* Legacy compatibility */

/*
 * Get effective max block size based on mode flags.
 * Killer mode allows larger cages to match Killer Sudoku conventions.
 */
static inline int get_maxblk(int mode_flags) {
    if (HAS_MODE(mode_flags, MODE_KILLER)) {
        return MAXBLK_KILLER;
    }
    return MAXBLK_STANDARD;
}

/*
 * Get minimum cage size for a mode.
 * Killer mode typically uses larger minimum cages (2-4 cells).
 */
static inline int get_minblk(int mode_flags) {
    if (HAS_MODE(mode_flags, MODE_KILLER)) {
        return 2;  /* Killer mode: minimum 2 cells per cage */
    }
    return 1;  /* Standard: allow singletons */
}

enum { COL_BACKGROUND, COL_GRID, COL_USER, COL_HIGHLIGHT, COL_ERROR, COL_PENCIL, NCOLOURS };

struct clues {
    int refcount;
    int w;
    int *dsf;
    long *clues;
};

struct game_state {
    game_params par;
    struct clues *clues;
    digit *grid;
    int *pencil; /* bitmaps using bits 1<<1..1<<n */
    int completed, cheated;
};

static game_params *default_params(void) {
    game_params *ret = snew(game_params);

    ret->w = 6;
    ret->diff = DIFF_NORMAL;
    ret->multiplication_only = FALSE;
    ret->mode_flags = MODE_STANDARD;

    return ret;
}

const static struct game_params keen_presets[] = {
    {3, DIFF_EASY, FALSE, MODE_STANDARD},
    {4, DIFF_EASY, FALSE, MODE_STANDARD},
    {5, DIFF_NORMAL, FALSE, MODE_STANDARD},
    {6, DIFF_NORMAL, FALSE, MODE_STANDARD},
    {7, DIFF_HARD, FALSE, MODE_STANDARD},
    {8, DIFF_EXTREME, FALSE, MODE_STANDARD},
    {9, DIFF_UNREASONABLE, FALSE, MODE_STANDARD},
};

static int game_fetch_preset(int i, char **name, game_params **params) {
    game_params *ret;
    char buf[80];

    if (i < 0 || i >= lenof(keen_presets))
        return FALSE;

    ret = snew(game_params);
    *ret = keen_presets[i]; /* structure copy */

    sprintf(buf, "%dx%d %s%s", ret->w, ret->w, keen_diffnames[ret->diff],
            ret->multiplication_only ? ", multiplication only" : "");

    *name = dupstr(buf);
    *params = ret;
    return TRUE;
}

static void free_params(game_params *params) {
    sfree(params);
}

static game_params *dup_params(const game_params *params) {
    game_params *ret = snew(game_params);
    *ret = *params; /* structure copy */
    return ret;
}

static void decode_params(game_params *params, char const *string) {
    char const *p = string;

    params->w = atoi(p);
    while (*p && isdigit((unsigned char)*p))
        p++;

    if (*p == 'd') {
        int i;
        p++;
        params->diff = DIFFCOUNT + 1; /* ...which is invalid */
        if (*p) {
            for (i = 0; i < DIFFCOUNT; i++) {
                if (*p == keen_diffchars[i])
                    params->diff = i;
            }
            p++;
        }
    }

    if (*p == 'm') {
        p++;
        params->multiplication_only = TRUE;
    }
}

static char *encode_params(const game_params *params, int full) {
    char ret[80];

    sprintf(ret, "%d", params->w);
    if (full)
        sprintf(ret + strlen(ret), "d%c%s", keen_diffchars[params->diff],
                params->multiplication_only ? "m" : "");

    return dupstr(ret);
}

static config_item *game_configure(const game_params *params) {
    config_item *ret;
    char buf[80];

    ret = snewn(4, config_item);

    ret[0].name = "Grid size";
    ret[0].type = C_STRING;
    sprintf(buf, "%d", params->w);
    ret[0].sval = dupstr(buf);
    ret[0].ival = 0;

    ret[1].name = "Difficulty";
    ret[1].type = C_CHOICES;
    ret[1].sval = DIFFCONFIG;
    ret[1].ival = params->diff;

    ret[2].name = "Multiplication only";
    ret[2].type = C_BOOLEAN;
    ret[2].sval = NULL;
    ret[2].ival = params->multiplication_only;

    ret[3].name = NULL;
    ret[3].type = C_END;
    ret[3].sval = NULL;
    ret[3].ival = 0;

    return ret;
}

static game_params *custom_params(const config_item *cfg) {
    game_params *ret = snew(game_params);

    ret->w = atoi(cfg[0].sval);
    ret->diff = cfg[1].ival;
    ret->multiplication_only = cfg[2].ival;
    ret->mode_flags = MODE_STANDARD;

    return ret;
}

static char *validate_params(const game_params *params, int full) {
    if (params->w < 3 || params->w > 16)
        return "Grid size must be between 3 and 16";
    if (params->diff >= DIFFCOUNT)
        return "Unknown difficulty rating";
    return NULL;
}

/* ----------------------------------------------------------------------
 * Solver.
 */

struct solver_ctx {
    int w, diff;
    int nboxes;
    int *boxes, *boxlist, *whichbox;
    long *clues;
    digit *soln;
    digit *dscratch;
    int *iscratch;
    int mode_flags;  /* Mode flags for Killer, Modular, etc. */
};

static void solver_clue_candidate(struct solver_ctx *ctx, int diff, int box) {
    int w = ctx->w;
    int n = ctx->boxes[box + 1] - ctx->boxes[box];
    int j;

    /*
     * Killer mode: reject candidates with duplicate digits in the cage
     */
    if (HAS_MODE(ctx->mode_flags, MODE_KILLER)) {
        int seen = 0;
        for (j = 0; j < n; j++) {
            int bit = 1 << ctx->dscratch[j];
            if (seen & bit)
                return;  /* Duplicate found, reject this candidate */
            seen |= bit;
        }
    }

    /*
     * This function is called from the main clue-based solver
     * routine when we discover a candidate layout for a given clue
     * box consistent with everything we currently know about the
     * digit constraints in that box. We expect to find the digits
     * of the candidate layout in ctx->dscratch, and we update
     * ctx->iscratch as appropriate.
     *
     * The contents of ctx->iscratch are completely different
     * depending on whether diff == DIFF_HARD or not. This function
     * uses iscratch completely differently between the two cases, and
     * the code in solver_common() which consumes the result must
     * likewise have an if statement with completely different
     * branches for the two cases.
     *
     * In DIFF_EASY and DIFF_NORMAL modes, the valid entries in
     * ctx->iscratch are 0,...,n-1, and each of those entries
     * ctx->iscratch[i] gives a bitmap of the possible digits in the
     * ith square of the clue box currently under consideration. So
     * each entry of iscratch starts off as an empty bitmap, and we
     * set bits in it as possible layouts for the clue box are
     * considered (and the difference between DIFF_EASY and
     * DIFF_NORMAL is just that in DIFF_EASY mode we deliberately set
     * more bits than absolutely necessary, hence restricting our own
     * knowledge).
     *
     * But in DIFF_HARD mode, the valid entries are 0,...,2*w-1 (at
     * least outside *this* function - inside this function, we also
     * use 2*w,...,4*w-1 as scratch space in the loop below); the
     * first w of those give the possible digits in the intersection
     * of the current clue box with each column of the puzzle, and the
     * next w do the same for each row. In this mode, each iscratch
     * entry starts off as a _full_ bitmap, and in this function we
     * _clear_ bits for digits that are absent from a given row or
     * column in each candidate layout, so that the only bits which
     * remain set are those for digits which have to appear in a given
     * row/column no matter how the clue box is laid out.
     */
    if (diff == DIFF_EASY) {
        unsigned mask = 0;
        /*
         * Easy-mode clue deductions: we do not record information
         * about which squares take which values, so we amalgamate
         * all the values in dscratch and OR them all into
         * everywhere.
         */
        for (j = 0; j < n; j++)
            mask |= 1 << ctx->dscratch[j];
        for (j = 0; j < n; j++)
            ctx->iscratch[j] |= mask;
    } else if (diff == DIFF_NORMAL) {
        /*
         * Normal-mode deductions: we process the information in
         * dscratch in the obvious way.
         */
        for (j = 0; j < n; j++)
            ctx->iscratch[j] |= 1 << ctx->dscratch[j];
    } else if (diff == DIFF_HARD) {
        /*
         * Hard-mode deductions: instead of ruling things out
         * _inside_ the clue box, we look for numbers which occur in
         * a given row or column in all candidate layouts, and rule
         * them out of all squares in that row or column that
         * _aren't_ part of this clue box.
         */
        int *sq = ctx->boxlist + ctx->boxes[box];

        for (j = 0; j < 2 * w; j++)
            ctx->iscratch[2 * w + j] = 0;
        for (j = 0; j < n; j++) {
            int x = sq[j] / w, y = sq[j] % w;
            ctx->iscratch[2 * w + x] |= 1 << ctx->dscratch[j];
            ctx->iscratch[3 * w + y] |= 1 << ctx->dscratch[j];
        }
        for (j = 0; j < 2 * w; j++)
            ctx->iscratch[j] &= ctx->iscratch[2 * w + j];
    }
}

static int solver_common(struct latin_solver *solver, void *vctx, int diff) {
    struct solver_ctx *ctx = (struct solver_ctx *)vctx;
    int w = ctx->w;
    int box, i, j, k;
    int ret = 0, total;

    /*
     * Iterate over each clue box and deduce what we can.
     */
    for (box = 0; box < ctx->nboxes; box++) {
        int *sq = ctx->boxlist + ctx->boxes[box];
        int n = ctx->boxes[box + 1] - ctx->boxes[box];
        long value = ctx->clues[box] & ~CMASK;
        long op = ctx->clues[box] & CMASK;

        /*
         * Initialise ctx->iscratch for this clue box. At different
         * difficulty levels we must initialise a different amount of
         * it to different things; see the comments in
         * solver_clue_candidate explaining what each version does.
         */
        if (diff == DIFF_HARD) {
            for (i = 0; i < 2 * w; i++)
                ctx->iscratch[i] = (1 << (w + 1)) - (1 << 1);
        } else {
            for (i = 0; i < n; i++)
                ctx->iscratch[i] = 0;
        }

        switch (op) {
        case C_SUB:
        case C_DIV:
            /*
             * These two clue types must always apply to a box of
             * area 2. Also, the two digits in these boxes can never
             * be the same (because any domino must have its two
             * squares in either the same row or the same column).
             * So we simply iterate over all possibilities for the
             * two squares (both ways round), rule out any which are
             * inconsistent with the digit constraints we already
             * have, and update the digit constraints with any new
             * information thus garnered.
             */
            assert(n == 2);

            for (i = 1; i <= w; i++) {
                j = (op == C_SUB ? i + value : i * value);
                if (j > w)
                    break;

                /* (i,j) is a valid digit pair. Try it both ways round. */

                if (solver->cube[sq[0] * w + i - 1] && solver->cube[sq[1] * w + j - 1]) {
                    ctx->dscratch[0] = i;
                    ctx->dscratch[1] = j;
                    solver_clue_candidate(ctx, diff, box);
                }

                if (solver->cube[sq[0] * w + j - 1] && solver->cube[sq[1] * w + i - 1]) {
                    ctx->dscratch[0] = j;
                    ctx->dscratch[1] = i;
                    solver_clue_candidate(ctx, diff, box);
                }
            }

            break;

        case C_EXP:
            /*
             * Exponentiation: base^exp = value. Only for 2-cell boxes.
             * Find all (base, exp) pairs where base^exp == clue value.
             * Convention: smaller digit is base, larger is exponent.
             */
            assert(n == 2);

            for (i = 1; i <= w; i++) {
                for (j = i + 1; j <= w; j++) {
                    /* Check if i^j == value */
                    long result = 1;
                    int e;
                    for (e = 0; e < j; e++) {
                        result *= i;
                        if (result > value)
                            break;  /* Early exit for overflow */
                    }
                    if (result != value)
                        continue;

                    /* (i,j) is base^exp = value. Try both cell orderings. */
                    if (solver->cube[sq[0] * w + i - 1] && solver->cube[sq[1] * w + j - 1]) {
                        ctx->dscratch[0] = i;
                        ctx->dscratch[1] = j;
                        solver_clue_candidate(ctx, diff, box);
                    }

                    if (solver->cube[sq[0] * w + j - 1] && solver->cube[sq[1] * w + i - 1]) {
                        ctx->dscratch[0] = j;
                        ctx->dscratch[1] = i;
                        solver_clue_candidate(ctx, diff, box);
                    }
                }
            }

            break;

        case C_ADD:
        case C_MUL:
            /*
             * Addition and multiplication cages require exhaustive
             * enumeration of all valid digit combinations.
             *
             * Rather than recursive function calls, we use iterative
             * enumeration via the scratch array. The index i tracks
             * which cell in the cage is currently being incremented.
             */
            i = 0;
            ctx->dscratch[i] = 0;
            total = value; /* start with the identity */
            while (1) {
                if (i < n) {
                    /*
                     * Find the next valid value for cell i.
                     */
                    for (j = ctx->dscratch[i] + 1; j <= w; j++) {
                        if (op == C_ADD ? (total < j) : (total % j != 0))
                            continue; /* this one won't fit */
                        if (!solver->cube[sq[i] * w + j - 1])
                            continue; /* this one is ruled out already */
                        for (k = 0; k < i; k++)
                            if (ctx->dscratch[k] == j &&
                                (sq[k] % w == sq[i] % w || sq[k] / w == sq[i] / w))
                                break; /* clashes with another row/col */
                        if (k < i)
                            continue;

                        /* Found one. */
                        break;
                    }

                    if (j > w) {
                        /* No valid values left; drop back. */
                        i--;
                        if (i < 0)
                            break; /* overall iteration is finished */
                        if (op == C_ADD)
                            total += ctx->dscratch[i];
                        else
                            total *= ctx->dscratch[i];
                    } else {
                        /* Got a valid value; store it and move on. */
                        ctx->dscratch[i++] = j;
                        if (op == C_ADD)
                            total -= j;
                        else
                            total /= j;
                        ctx->dscratch[i] = 0;
                    }
                } else {
                    if (total == (op == C_ADD ? 0 : 1))
                        solver_clue_candidate(ctx, diff, box);
                    i--;
                    if (op == C_ADD)
                        total += ctx->dscratch[i];
                    else
                        total *= ctx->dscratch[i];
                }
            }

            break;

        case C_MOD:
            /*
             * Modulo operation: a % b = clue. 2-cell cages only.
             * For each divisor d from 1 to w, find all pairs (a,d) where a % d = value.
             * Since we want pairs where the larger mod smaller = value:
             *   dividend % divisor = remainder  =>  dividend = k*divisor + remainder
             */
            assert(n == 2);

            for (i = 1; i <= w; i++) {  /* divisor */
                if (value >= i) continue;  /* remainder must be < divisor */
                for (j = 1; j <= w; j++) {  /* dividend */
                    if (j <= i) continue;  /* dividend > divisor for meaningful mod */
                    if (j % i != value) continue;

                    /* (j, i) is a valid pair: j % i = value */
                    if (solver->cube[sq[0] * w + j - 1] && solver->cube[sq[1] * w + i - 1]) {
                        ctx->dscratch[0] = j;
                        ctx->dscratch[1] = i;
                        solver_clue_candidate(ctx, diff, box);
                    }
                    if (solver->cube[sq[0] * w + i - 1] && solver->cube[sq[1] * w + j - 1]) {
                        ctx->dscratch[0] = i;
                        ctx->dscratch[1] = j;
                        solver_clue_candidate(ctx, diff, box);
                    }
                }
            }
            break;

        case C_GCD:
            /*
             * GCD cages: GCD of all digits = clue value.
             * Uses iterative enumeration similar to ADD/MUL.
             * All digits must be divisible by the clue (GCD result).
             */
            i = 0;
            ctx->dscratch[i] = 0;
            while (1) {
                if (i < n) {
                    /* Find next digit that is divisible by the target GCD */
                    for (j = ctx->dscratch[i] + 1; j <= w; j++) {
                        if (j % value != 0)
                            continue;  /* Must be multiple of GCD */
                        if (!solver->cube[sq[i] * w + j - 1])
                            continue;
                        /* Check Latin square constraint */
                        for (k = 0; k < i; k++)
                            if (ctx->dscratch[k] == j &&
                                (sq[k] % w == sq[i] % w || sq[k] / w == sq[i] / w))
                                break;
                        if (k < i)
                            continue;
                        break;
                    }

                    if (j > w) {
                        i--;
                        if (i < 0) break;
                    } else {
                        ctx->dscratch[i++] = j;
                        ctx->dscratch[i] = 0;
                    }
                } else {
                    /* Check if GCD of collected digits equals value */
                    if (gcd_array(ctx->dscratch, n) == value)
                        solver_clue_candidate(ctx, diff, box);
                    i--;
                }
            }
            break;

        case C_LCM:
            /*
             * LCM cages: LCM of all digits = clue value.
             * All digits must be divisors of the clue value.
             */
            i = 0;
            ctx->dscratch[i] = 0;
            while (1) {
                if (i < n) {
                    /* Find next digit that divides the target LCM */
                    for (j = ctx->dscratch[i] + 1; j <= w; j++) {
                        if (value % j != 0)
                            continue;  /* Must divide LCM */
                        if (!solver->cube[sq[i] * w + j - 1])
                            continue;
                        /* Check Latin square constraint */
                        for (k = 0; k < i; k++)
                            if (ctx->dscratch[k] == j &&
                                (sq[k] % w == sq[i] % w || sq[k] / w == sq[i] / w))
                                break;
                        if (k < i)
                            continue;
                        break;
                    }

                    if (j > w) {
                        i--;
                        if (i < 0) break;
                    } else {
                        ctx->dscratch[i++] = j;
                        ctx->dscratch[i] = 0;
                    }
                } else {
                    /* Check if LCM of collected digits equals value */
                    if (lcm_array(ctx->dscratch, n) == value)
                        solver_clue_candidate(ctx, diff, box);
                    i--;
                }
            }
            break;
        }

        /*
         * Do deductions based on the information we've now
         * accumulated in ctx->iscratch. See the comments above in
         * solver_clue_candidate explaining what data is left in here,
         * and how it differs between DIFF_HARD and lower difficulty
         * levels (hence the big if statement here).
         */
        if (diff < DIFF_HARD) {
#ifdef STANDALONE_SOLVER
            char prefix[256];

            if (solver_show_working)
                sprintf(prefix, "%*susing clue at (%d,%d):\n", solver_recurse_depth * 4, "",
                        sq[0] / w + 1, sq[0] % w + 1);
            else
                prefix[0] = '\0'; /* placate optimiser */
#endif

            for (i = 0; i < n; i++)
                for (j = 1; j <= w; j++) {
                    if (solver->cube[sq[i] * w + j - 1] && !(ctx->iscratch[i] & (1 << j))) {
#ifdef STANDALONE_SOLVER
                        if (solver_show_working) {
                            printf("%s%*s  ruling out %d at (%d,%d)\n", prefix,
                                   solver_recurse_depth * 4, "", j, sq[i] / w + 1, sq[i] % w + 1);
                            prefix[0] = '\0';
                        }
#endif
                        solver->cube[sq[i] * w + j - 1] = 0;
                        ret = 1;
                    }
                }
        } else {
#ifdef STANDALONE_SOLVER
            char prefix[256];

            if (solver_show_working)
                sprintf(prefix, "%*susing clue at (%d,%d):\n", solver_recurse_depth * 4, "",
                        sq[0] / w + 1, sq[0] % w + 1);
            else
                prefix[0] = '\0'; /* placate optimiser */
#endif

            for (i = 0; i < 2 * w; i++) {
                int start = (i < w ? i * w : i - w);
                int step = (i < w ? 1 : w);
                for (j = 1; j <= w; j++)
                    if (ctx->iscratch[i] & (1 << j)) {
#ifdef STANDALONE_SOLVER
                        char prefix2[256];

                        if (solver_show_working)
                            sprintf(prefix2,
                                    "%*s  this clue requires %d in"
                                    " %s %d:\n",
                                    solver_recurse_depth * 4, "", j, i < w ? "column" : "row",
                                    i % w + 1);
                        else
                            prefix2[0] = '\0'; /* placate optimiser */
#endif

                        for (k = 0; k < w; k++) {
                            int pos = start + k * step;
                            if (ctx->whichbox[pos] != box && solver->cube[pos * w + j - 1]) {
#ifdef STANDALONE_SOLVER
                                if (solver_show_working) {
                                    printf("%s%s%*s   ruling out %d at (%d,%d)\n", prefix, prefix2,
                                           solver_recurse_depth * 4, "", j, pos / w + 1,
                                           pos % w + 1);
                                    prefix[0] = prefix2[0] = '\0';
                                }
#endif
                                solver->cube[pos * w + j - 1] = 0;
                                ret = 1;
                            }
                        }
                    }
            }

            /*
             * Once we find one block we can do something with in
             * this way, revert to trying easier deductions, so as
             * not to generate solver diagnostics that make the
             * problem look harder than it is. (We have to do this
             * for the Hard deductions but not the Easy/Normal ones,
             * because only the Hard deductions are cross-box.)
             */
            if (ret)
                return ret;
        }
    }

    return ret;
}

static int solver_easy(struct latin_solver *solver, void *vctx) {
    /*
     * Omit the EASY deductions when solving at NORMAL level, since
     * the NORMAL deductions are a superset of them anyway and it
     * saves on time and confusing solver diagnostics.
     *
     * Note that this breaks the natural semantics of the return
     * value of latin_solver. Without this hack, you could determine
     * a puzzle's difficulty in one go by trying to solve it at
     * maximum difficulty and seeing what difficulty value was
     * returned; but with this hack, solving an Easy puzzle on
     * Normal difficulty will typically return Normal. Hence the
     * uses of the solver to determine difficulty are all arranged
     * so as to double-check by re-solving at the next difficulty
     * level down and making sure it failed.
     */
    struct solver_ctx *ctx = (struct solver_ctx *)vctx;
    if (ctx->diff > DIFF_EASY)
        return 0;
    return solver_common(solver, vctx, DIFF_EASY);
}

static int solver_normal(struct latin_solver *solver, void *vctx) {
    return solver_common(solver, vctx, DIFF_NORMAL);
}

static int solver_hard(struct latin_solver *solver, void *vctx) {
    return solver_common(solver, vctx, DIFF_HARD);
}

#define SOLVER(upper, title, func, lower) func,
static usersolver_t const keen_solvers[] = {DIFFLIST(SOLVER)};

static int solver(int w, int *dsf, long *clues, digit *soln, int maxdiff, int mode_flags) {
    int a = w * w;
    struct solver_ctx ctx;
    int ret;
    int i, j, n, m;

    ctx.w = w;
    ctx.soln = soln;
    ctx.diff = maxdiff;
    ctx.mode_flags = mode_flags;

    /*
     * Transform the dsf-formatted clue list into one over which we
     * can iterate more easily.
     *
     * Also transpose the x- and y-coordinates at this point,
     * because the 'cube' array in the general Latin square solver
     * puts x first (oops).
     */
    for (ctx.nboxes = i = 0; i < a; i++)
        if (dsf_canonify(dsf, i) == i)
            ctx.nboxes++;
    ctx.boxlist = snewn(a, int);
    ctx.boxes = snewn(ctx.nboxes + 1, int);
    ctx.clues = snewn(ctx.nboxes, long);
    ctx.whichbox = snewn(a, int);
    for (n = m = i = 0; i < a; i++)
        if (dsf_canonify(dsf, i) == i) {
            ctx.clues[n] = clues[i];
            ctx.boxes[n] = m;
            for (j = 0; j < a; j++)
                if (dsf_canonify(dsf, j) == i) {
                    ctx.boxlist[m++] = (j % w) * w + (j / w); /* transpose */
                    ctx.whichbox[ctx.boxlist[m - 1]] = n;
                }
            n++;
        }
    assert(n == ctx.nboxes);
    assert(m == a);
    ctx.boxes[n] = m;

    ctx.dscratch = snewn(a + 1, digit);
    ctx.iscratch = snewn(max(a + 1, 4 * w), int);

    ret = latin_solver(soln, w, maxdiff, DIFF_EASY, DIFF_HARD, DIFF_EXTREME, DIFF_EXTREME,
                       DIFF_UNREASONABLE, keen_solvers, &ctx, NULL, NULL);

    sfree(ctx.dscratch);
    sfree(ctx.iscratch);
    sfree(ctx.whichbox);
    sfree(ctx.boxlist);
    sfree(ctx.boxes);
    sfree(ctx.clues);

    return ret;
}

/* ----------------------------------------------------------------------
 * Grid generation.
 */

static char *encode_block_structure(char *p, int w, int *dsf) {
    int i, currrun = 0;
    char *orig, *q, *r, c;

    orig = p;

    /*
     * Encode the block structure. We do this by encoding the
     * pattern of dividing lines: first we iterate over the w*(w-1)
     * internal vertical grid lines in ordinary reading order, then
     * over the w*(w-1) internal horizontal ones in transposed
     * reading order.
     *
     * We encode the number of non-lines between the lines; _ means
     * zero (two adjacent divisions), a means 1, ..., y means 25,
     * and z means 25 non-lines _and no following line_ (so that za
     * means 26, zb 27 etc).
     */
    for (i = 0; i <= 2 * w * (w - 1); i++) {
        int x, y, p0, p1, edge;

        if (i == 2 * w * (w - 1)) {
            edge = TRUE; /* terminating virtual edge */
        } else {
            if (i < w * (w - 1)) {
                y = i / (w - 1);
                x = i % (w - 1);
                p0 = y * w + x;
                p1 = y * w + x + 1;
            } else {
                x = i / (w - 1) - w;
                y = i % (w - 1);
                p0 = y * w + x;
                p1 = (y + 1) * w + x;
            }
            edge = (dsf_canonify(dsf, p0) != dsf_canonify(dsf, p1));
        }

        if (edge) {
            while (currrun > 25)
                *p++ = 'z', currrun -= 25;
            if (currrun)
                *p++ = 'a' - 1 + currrun;
            else
                *p++ = '_';
            currrun = 0;
        } else
            currrun++;
    }

    /*
     * Now go through and compress the string by replacing runs of
     * the same letter with a single copy of that letter followed by
     * a repeat count, where that makes it shorter. (This puzzle
     * seems to generate enough long strings of _ to make this a
     * worthwhile step.)
     */
    for (q = r = orig; r < p;) {
        *q++ = c = *r;

        for (i = 0; r + i < p && r[i] == c; i++)
            ;
        r += i;

        if (i == 2) {
            *q++ = c;
        } else if (i > 2) {
            q += sprintf(q, "%d", i);
        }
    }

    return q;
}

static char *parse_block_structure(const char **p, int w, int *dsf) {
    int a = w * w;
    int pos = 0;
    int repc = 0, repn = 0;

    dsf_init(dsf, a);

    while (**p && (repn > 0 || **p != ',')) {
        int c, adv;

        if (repn > 0) {
            repn--;
            c = repc;
        } else if (**p == '_' || (**p >= 'a' && **p <= 'z')) {
            c = (**p == '_' ? 0 : **p - 'a' + 1);
            (*p)++;
            if (**p && isdigit((unsigned char)**p)) {
                repc = c;
                repn = atoi(*p) - 1;
                while (**p && isdigit((unsigned char)**p))
                    (*p)++;
            }
        } else
            return "Invalid character in game description";

        adv = (c != 25); /* 'z' is a special case */

        while (c-- > 0) {
            int p0, p1;

            /*
             * Non-edge; merge the two dsf classes on either
             * side of it.
             */
            if (pos >= 2 * w * (w - 1))
                return "Too much data in block structure specification";
            if (pos < w * (w - 1)) {
                int y = pos / (w - 1);
                int x = pos % (w - 1);
                p0 = y * w + x;
                p1 = y * w + x + 1;
            } else {
                int x = pos / (w - 1) - w;
                int y = pos % (w - 1);
                p0 = y * w + x;
                p1 = (y + 1) * w + x;
            }
            dsf_merge(dsf, p0, p1);

            pos++;
        }
        if (adv) {
            pos++;
            if (pos > 2 * w * (w - 1) + 1)
                return "Too much data in block structure specification";
        }
    }

    /*
     * When desc is exhausted, we expect to have gone exactly
     * one space _past_ the end of the grid, due to the dummy
     * edge at the end.
     */
    if (pos != 2 * w * (w - 1) + 1)
        return "Not enough data in block structure specification";

    return NULL;
}

char *new_game_desc(const game_params *params, random_state *rs, char **aux, int interactive) {
    int w = params->w, a = w * w;
    digit *grid, *soln;
    int *order, *revorder, *singletons, *dsf;
    long *clues, *cluevals;
    int i, j, k, n, x, y, ret;
    int diff = params->diff;
    int maxblk = get_maxblk(params->mode_flags);
    (void)get_minblk(params->mode_flags);  /* Reserved for future constraint validation */
    char *desc, *p;

    /*
     * Legacy difficulty exception for 3x3 grids.
     *
     * Note: With AI-assisted generation (NeuralKenKenGenerator), this
     * limitation is bypassed since the neural model provides valid
     * Latin squares directly, avoiding the exhaustive search that
     * caused timeouts in pure algorithmic generation.
     *
     * This fallback remains for pure C generation path only.
     *
     * TODO(expansion): Consider removing this entirely once AI generation
     * is the primary path, or add similar safeguards for 4x4 Extreme/
     * Unreasonable if needed.
     */
    if (w == 3 && diff > DIFF_NORMAL)
        diff = DIFF_NORMAL;

    grid = NULL;

    order = snewn(a, int);
    revorder = snewn(a, int);
    singletons = snewn(a, int);
    dsf = snew_dsf(a);
    clues = snewn(a, long);
    cluevals = snewn(a, long);
    soln = snewn(a, digit);

    /*
     * Limit retries to prevent infinite loops when mode constraints or
     * large grid sizes make valid puzzles rare. Scale with grid size
     * since larger grids naturally need more attempts.
     */
    int max_retries = 1000 + (a * 10);  /* Base + size-scaled */
    int attempts = 0;

    while (attempts < max_retries) {
        attempts++;
        /*
         * First construct a latin square to be the solution.
         */
        sfree(grid);
        grid = latin_generate(w, rs);

        /*
         * Zero-Inclusive mode: convert digits from 1..N to 0..(N-1)
         */
        if (HAS_MODE(params->mode_flags, MODE_ZERO_INCLUSIVE)) {
            for (i = 0; i < a; i++) {
                grid[i] -= 1;
            }
        }

        /*
         * Negative Numbers mode: convert digits from 1..N to -floor(N/2)..-1, 1..ceil(N/2)
         * Example for N=5: 1,2,3,4,5 -> -2,-1,1,2,3
         * Example for N=6: 1,2,3,4,5,6 -> -3,-2,-1,1,2,3
         */
        if (HAS_MODE(params->mode_flags, MODE_NEGATIVE)) {
            int neg_count = w / 2;
            for (i = 0; i < a; i++) {
                if (grid[i] <= neg_count)
                    grid[i] = grid[i] - neg_count - 1;  /* Maps to -neg_count..-1 */
                else
                    grid[i] = grid[i] - neg_count;      /* Maps to 1..(N-neg_count) */
            }
        }

        /*
         * Divide the grid into arbitrarily sized blocks, but so as
         * to arrange plenty of dominoes which can be SUB/DIV clues.
         * We do this by first placing dominoes at random for a
         * while, then tying the remaining singletons one by one
         * into neighbouring blocks.
         */
        for (i = 0; i < a; i++)
            order[i] = i;

        shuffle(order, a, sizeof(*order), rs);

        for (i = 0; i < a; i++)
            revorder[order[i]] = i;

        for (i = 0; i < a; i++)
            singletons[i] = TRUE;

        dsf_init(dsf, a);

        /* Place dominoes. */
        for (i = 0; i < a; i++) {
            if (singletons[i]) {
                int best = -1;

                x = i % w;
                y = i / w;

                if (x > 0 && singletons[i - 1] && (best == -1 || revorder[i - 1] < revorder[best]))
                    best = i - 1;

                if (x + 1 < w && singletons[i + 1] &&
                    (best == -1 || revorder[i + 1] < revorder[best]))
                    best = i + 1;

                if (y > 0 && singletons[i - w] && (best == -1 || revorder[i - w] < revorder[best]))
                    best = i - w;

                if (y + 1 < w && singletons[i + w] &&
                    (best == -1 || revorder[i + w] < revorder[best]))
                    best = i + w;

                /*
                 * When we find a potential domino, we place it with
                 * probability 3/4, which seems to strike a decent
                 * balance between plenty of dominoes and leaving
                 * enough singletons to make interesting larger
                 * shapes.
                 */
                if (best >= 0 && random_upto(rs, 4)) {
                    singletons[i] = singletons[best] = FALSE;
                    dsf_merge(dsf, i, best);
                }
            }
        }

        /* Fold in singletons. */
        for (i = 0; i < a; i++) {
            if (singletons[i]) {
                int best = -1;

                x = i % w;
                y = i / w;

                if (x > 0 && dsf_size(dsf, i - 1) < maxblk &&
                    (best == -1 || revorder[i - 1] < revorder[best]))
                    best = i - 1;

                if (x + 1 < w && dsf_size(dsf, i + 1) < maxblk &&
                    (best == -1 || revorder[i + 1] < revorder[best]))
                    best = i + 1;

                if (y > 0 && dsf_size(dsf, i - w) < maxblk &&
                    (best == -1 || revorder[i - w] < revorder[best]))
                    best = i - w;

                if (y + 1 < w && dsf_size(dsf, i + w) < maxblk &&
                    (best == -1 || revorder[i + w] < revorder[best]))
                    best = i + w;

                if (best >= 0) {
                    singletons[i] = singletons[best] = FALSE;
                    dsf_merge(dsf, i, best);
                }
            }
        }

        /* Quit and start again if we have any singletons left over
         * which we weren't able to do anything at all with. */
        for (i = 0; i < a; i++)
            if (singletons[i])
                break;

        if (i < a)
            continue;

        /*
         * Decide what would be acceptable clues for each block.
         *
         * Blocks larger than 2 have free choice of ADD or MUL;
         * blocks of size 2 can be anything in principle (except
         * that they can only be DIV if the two numbers have an
         * integer quotient, of course), but we rule out (or try to
         * avoid) some clues because they're of low quality.
         *
         * Hence, we iterate once over the grid, stopping at the
         * canonical element of every >2 block and the _non_-
         * canonical element of every 2-block; the latter means that
         * we can make our decision about a 2-block in the knowledge
         * of both numbers in it.
         *
         * We reuse the 'singletons' array (finished with in the
         * above loop) to hold information about which blocks are
         * suitable for what.
         */
#define F_ADD 0x001
#define F_SUB 0x002
#define F_MUL 0x004
#define F_DIV 0x008
#define F_EXP 0x010  /* Exponentiation: base^exp */
#define F_MOD 0x020  /* Modulo: larger % smaller */
#define F_GCD 0x040  /* Greatest common divisor */
#define F_LCM 0x080  /* Least common multiple */
#define BAD_SHIFT 8  /* Increased for 8 clue types */

        for (i = 0; i < a; i++) {
            singletons[i] = 0;
            j = dsf_canonify(dsf, i);
            k = dsf_size(dsf, j);
            if (params->multiplication_only)
                singletons[j] = F_MUL;
            else if (j == i && k > 2) {
                singletons[j] |= F_ADD | F_MUL;
            } else if (j != i && k == 2) {
                /* Fetch the two numbers and sort them into order. */
                int p = grid[j], q = grid[i], v;
                if (p < q) {
                    int t = p;
                    p = q;
                    q = t;
                }

                /*
                 * Addition clues are always allowed, but we try to
                 * avoid sums of 3, 4, (2w-1) and (2w-2) if we can,
                 * because they're too easy - they only leave one
                 * option for the pair of numbers involved.
                 */
                v = p + q;
                if (v > 4 && v < 2 * w - 2)
                    singletons[j] |= F_ADD;
                else
                    singletons[j] |= F_ADD << BAD_SHIFT;

                /*
                 * Multiplication clues: above Normal difficulty, we
                 * prefer (but don't absolutely insist on) clues of
                 * this type which leave multiple options open.
                 */
                v = p * q;
                n = 0;
                for (k = 1; k <= w; k++)
                    if (v % k == 0 && v / k <= w && v / k != k)
                        n++;
                if (n <= 2 && diff > DIFF_NORMAL)
                    singletons[j] |= F_MUL << BAD_SHIFT;
                else
                    singletons[j] |= F_MUL;

                /*
                 * Subtraction: we completely avoid a difference of
                 * w-1.
                 */
                v = p - q;
                if (v < w - 1)
                    singletons[j] |= F_SUB;

                /*
                 * Division: for a start, the quotient must be an
                 * integer or the clue type is impossible. Also, we
                 * never use quotients strictly greater than w/2,
                 * because they're not only too easy but also
                 * inelegant.
                 *
                 * Zero-Inclusive mode: division is disabled entirely
                 * because 0 as a divisor creates ambiguity.
                 * Negative mode: disabled due to sign ambiguity in quotients.
                 * Modular mode: disabled (modular division is complex).
                 */
                if (!HAS_MODE(params->mode_flags, MODE_ZERO_INCLUSIVE) &&
                    !HAS_MODE(params->mode_flags, MODE_NEGATIVE) &&
                    !HAS_MODE(params->mode_flags, MODE_MODULAR) &&
                    p % q == 0 && 2 * (p / q) <= w)
                    singletons[j] |= F_DIV;

                /*
                 * Exponentiation: only for 2-cell cages when MODE_EXPONENT
                 * is active. We use base^exp where result <= reasonable max.
                 * Only allow when q >= 2 to avoid trivial x^1 = x cases.
                 */
                if (HAS_MODE(params->mode_flags, MODE_EXPONENT) && q >= 2) {
                    /* Check if p^q or q^p yields a reasonable clue value */
                    long exp_val = 1;
                    int valid = 1;
                    for (int e = 0; e < q && valid; e++) {
                        exp_val *= p;
                        if (exp_val > 10000) valid = 0;  /* Clue too large */
                    }
                    if (valid && exp_val <= 10000)
                        singletons[j] |= F_EXP;
                }

                /*
                 * Number theory operations: MOD, GCD, LCM
                 * Only available when MODE_NUMBER_THEORY is active.
                 */
                if (HAS_MODE(params->mode_flags, MODE_NUMBER_THEORY)) {
                    /*
                     * Modulo: p % q (larger % smaller). Avoid trivial cases
                     * where remainder is 0 (that's just division info).
                     */
                    v = p % q;
                    if (v > 0 && v < q)
                        singletons[j] |= F_MOD;

                    /*
                     * GCD: gcd(p, q). Most interesting when result is
                     * neither 1 (coprime, trivial) nor max(p,q) (one divides other).
                     */
                    v = (int)gcd_helper(p, q);
                    if (v > 1 && v < q)
                        singletons[j] |= F_GCD;
                    else if (v == 1)
                        singletons[j] |= F_GCD << BAD_SHIFT;  /* Less interesting */

                    /*
                     * LCM: lcm(p, q). Avoid cases where LCM > 100 (clue overflow)
                     * or LCM = p*q (coprime, trivial).
                     */
                    v = (int)lcm_helper(p, q);
                    if (v <= 100 && v < p * q)
                        singletons[j] |= F_LCM;
                    else if (v <= 100)
                        singletons[j] |= F_LCM << BAD_SHIFT;
                }
            }
        }

        /*
         * Assign clue types to blocks, balancing the distribution
         * across operation types and preferring optimal candidates.
         *
         * Note: O(N^2) complexity is acceptable here since N is bounded
         * by the maximum number of dominoes in a 9x9 grid.
         */
        shuffle(order, a, sizeof(*order), rs);
        for (i = 0; i < a; i++)
            clues[i] = 0;
        while (1) {
            int done_something = FALSE;

            for (k = 0; k < 8; k++) {
                long clue;
                int good, bad;
                switch (k) {
                case 0:
                    clue = C_DIV;
                    good = F_DIV;
                    break;
                case 1:
                    clue = C_SUB;
                    good = F_SUB;
                    break;
                case 2:
                    clue = C_MUL;
                    good = F_MUL;
                    break;
                case 3:
                    clue = C_ADD;
                    good = F_ADD;
                    break;
                case 4:
                    clue = C_EXP;
                    good = F_EXP;
                    break;
                case 5:
                    clue = C_MOD;
                    good = F_MOD;
                    break;
                case 6:
                    clue = C_GCD;
                    good = F_GCD;
                    break;
                case 7:
                    clue = C_LCM;
                    good = F_LCM;
                    break;
                default:
                    continue;  /* Safety fallback */
                }

                for (i = 0; i < a; i++) {
                    j = order[i];
                    if (singletons[j] & good) {
                        clues[j] = clue;
                        singletons[j] = 0;
                        break;
                    }
                }
                if (i == a) {
                    /* didn't find a nice one, use a nasty one */
                    bad = good << BAD_SHIFT;
                    for (i = 0; i < a; i++) {
                        j = order[i];
                        if (singletons[j] & bad) {
                            clues[j] = clue;
                            singletons[j] = 0;
                            break;
                        }
                    }
                }
                if (i < a)
                    done_something = TRUE;
            }

            if (!done_something)
                break;
        }
#undef F_ADD
#undef F_SUB
#undef F_MUL
#undef F_DIV
#undef F_EXP
#undef F_MOD
#undef F_GCD
#undef F_LCM
#undef BAD_SHIFT

        /*
         * Having chosen the clue types, calculate the clue values.
         */
        for (i = 0; i < a; i++) {
            j = dsf_canonify(dsf, i);
            if (j == i) {
                cluevals[j] = grid[i];
            } else {
                switch (clues[j]) {
                case C_ADD:
                    cluevals[j] += grid[i];
                    break;
                case C_MUL:
                    cluevals[j] *= grid[i];
                    break;
                case C_SUB:
                    cluevals[j] = labs(cluevals[j] - grid[i]);
                    break;
                case C_DIV: {
                    int d1 = cluevals[j], d2 = grid[i];
                    if (d1 == 0 || d2 == 0)
                        cluevals[j] = 0;
                    else
                        cluevals[j] = d2 / d1 + d1 / d2; /* one is 0 :-) */
                } break;
                case C_EXP: {
                    /* Exponentiation: smaller^larger (base^exp) */
                    long base, exp_val;
                    long result = 1;
                    int e;
                    if (cluevals[j] <= grid[i]) {
                        base = cluevals[j];
                        exp_val = grid[i];
                    } else {
                        base = grid[i];
                        exp_val = cluevals[j];
                    }
                    for (e = 0; e < exp_val; e++)
                        result *= base;
                    cluevals[j] = result;
                } break;
                case C_MOD: {
                    /* Modulo: larger % smaller (2-cell only) */
                    long larger = cluevals[j] > grid[i] ? cluevals[j] : grid[i];
                    long smaller = cluevals[j] <= grid[i] ? cluevals[j] : grid[i];
                    cluevals[j] = smaller > 0 ? larger % smaller : 0;
                } break;
                case C_GCD:
                    /* GCD: accumulate gcd(current, next) */
                    cluevals[j] = gcd_helper(cluevals[j], grid[i]);
                    break;
                case C_LCM:
                    /* LCM: accumulate lcm(current, next) */
                    cluevals[j] = lcm_helper(cluevals[j], grid[i]);
                    break;
                }
            }
        }

        /*
         * Modular mode: apply mod N to all clue values, keeping in range 1..N
         */
        if (HAS_MODE(params->mode_flags, MODE_MODULAR)) {
            for (i = 0; i < a; i++) {
                j = dsf_canonify(dsf, i);
                if (j == i) {
                    /* Map to 1..N range: ((val-1) mod N) + 1 */
                    cluevals[j] = ((cluevals[j] - 1) % w + w) % w + 1;
                }
            }
        }

        for (i = 0; i < a; i++) {
            j = dsf_canonify(dsf, i);
            if (j == i) {
                clues[j] |= cluevals[j];
            }
        }

        /*
         * See if the game can be solved at the specified difficulty
         * level, but not at the one below.
         */
        if (diff > 0) {
            memset(soln, 0, a);
            ret = solver(w, dsf, clues, soln, diff - 1, params->mode_flags);
            if (ret <= diff - 1)
                continue;
        }
        memset(soln, 0, a);
        ret = solver(w, dsf, clues, soln, diff, params->mode_flags);
        if (ret != diff)
            continue; /* go round again */

        /*
         * Potential optimization (not implemented): merge adjacent
         * blocks to increase difficulty when the puzzle is easier
         * than requested. This could improve generation success rate.
         * Currently omitted as generation speed is acceptable.
         */

        /*
         * We've got a usable puzzle!
         */
        break;
    }

    /*
     * Check if we exhausted retries without finding a valid puzzle.
     * This prevents hangs with problematic mode/size combinations.
     */
    if (attempts >= max_retries) {
        sfree(grid);
        sfree(order);
        sfree(revorder);
        sfree(singletons);
        sfree(dsf);
        sfree(clues);
        sfree(cluevals);
        sfree(soln);
        return NULL;
    }

    /*
     * Encode the puzzle description.
     */
    desc = snewn(40 * a, char);
    p = desc;
    // p = encode_block_structure(p, w, dsf);

    for (i = 0; i < a; i++) {
        j = dsf_canonify(dsf, i);
        p += sprintf(p, "%02d", j);
        if (i < a - 1)
            *p++ = ',';
    }

    *p++ = ';';

    for (i = 0; i < a; i++) {
        j = dsf_canonify(dsf, i);
        if (j == i) {
            switch (clues[j] & CMASK) {
            case C_ADD:
                *p++ = 'a';
                break;
            case C_SUB:
                *p++ = 's';
                break;
            case C_MUL:
                *p++ = 'm';
                break;
            case C_DIV:
                *p++ = 'd';
                break;
            case C_EXP:
                *p++ = 'e';
                break;
            case C_MOD:
                *p++ = 'o';  /* 'o' for m'o'dulo, 'm' taken by MUL */
                break;
            case C_GCD:
                *p++ = 'g';
                break;
            case C_LCM:
                *p++ = 'l';
                break;
            }
            p += sprintf(p, "%05ld", clues[j] & ~CMASK);

            *p++ = ',';
        }
    }

    --p;
    *p++ = '\0';

    desc = sresize(desc, p - desc, char);

    /*
     * Encode the solution.
     */
    assert(memcmp(soln, grid, a) == 0);
    *aux = snewn(a + 2, char);

    for (i = 0; i < a; i++)
        (*aux)[i] = '0' + soln[i];
    (*aux)[a] = 'e';
    (*aux)[a + 1] = '\0';

    sfree(grid);
    sfree(order);
    sfree(revorder);
    sfree(singletons);
    sfree(dsf);
    sfree(clues);
    sfree(cluevals);
    sfree(soln);

    return desc;
}

char *new_game_desc_from_grid(const game_params *params, random_state *rs, digit *input_grid,
                              char **aux, int interactive) {
    int w = params->w, a = w * w;
    digit *grid, *soln;
    int *order, *revorder, *singletons, *dsf;
    long *clues, *cluevals;
    int i, j, k, n, x, y, ret;
    int diff = params->diff;
    int maxblk = get_maxblk(params->mode_flags);
    (void)get_minblk(params->mode_flags);  /* Reserved for future constraint validation */
    char *desc, *p;

    /*
     * Difficulty exceptions: 3x3 puzzles at difficulty Hard or
     * higher are currently not generable.
     */
    if (w == 3 && diff > DIFF_NORMAL)
        diff = DIFF_NORMAL;

    grid = NULL;

    order = snewn(a, int);
    revorder = snewn(a, int);
    singletons = snewn(a, int);
    dsf = snew_dsf(a);
    clues = snewn(a, long);
    cluevals = snewn(a, long);
    soln = snewn(a, digit);

    /* Use the provided grid */
    grid = snewn(a, digit);
    memcpy(grid, input_grid, a * sizeof(digit));

    /*
     * Zero-Inclusive mode: AI model generates 1..N values,
     * convert them to 0..(N-1) for this mode.
     */
    if (HAS_MODE(params->mode_flags, MODE_ZERO_INCLUSIVE)) {
        for (i = 0; i < a; i++) {
            grid[i] -= 1;
        }
    }

    /*
     * Negative Numbers mode: convert digits from 1..N to -floor(N/2)..-1, 1..ceil(N/2)
     */
    if (HAS_MODE(params->mode_flags, MODE_NEGATIVE)) {
        int neg_count = w / 2;
        for (i = 0; i < a; i++) {
            if (grid[i] <= neg_count)
                grid[i] = grid[i] - neg_count - 1;
            else
                grid[i] = grid[i] - neg_count;
        }
    }

    /* Limit retries to prevent infinite loops with bad grids */
    int max_retries = 1000;
    int attempts = 0;

    while (attempts < max_retries) {
        attempts++;

        /*
         * Divide the grid into arbitrarily sized blocks.
         */
        for (i = 0; i < a; i++)
            order[i] = i;

        shuffle(order, a, sizeof(*order), rs);

        for (i = 0; i < a; i++)
            revorder[order[i]] = i;

        for (i = 0; i < a; i++)
            singletons[i] = TRUE;

        dsf_init(dsf, a);

        /* Place dominoes. */
        for (i = 0; i < a; i++) {
            if (singletons[i]) {
                int best = -1;

                x = i % w;
                y = i / w;

                if (x > 0 && singletons[i - 1] && (best == -1 || revorder[i - 1] < revorder[best]))
                    best = i - 1;

                if (x + 1 < w && singletons[i + 1] &&
                    (best == -1 || revorder[i + 1] < revorder[best]))
                    best = i + 1;

                if (y > 0 && singletons[i - w] && (best == -1 || revorder[i - w] < revorder[best]))
                    best = i - w;

                if (y + 1 < w && singletons[i + w] &&
                    (best == -1 || revorder[i + w] < revorder[best]))
                    best = i + w;

                if (best >= 0 && random_upto(rs, 4)) {
                    singletons[i] = singletons[best] = FALSE;
                    dsf_merge(dsf, i, best);
                }
            }
        }

        /* Fold in singletons. */
        for (i = 0; i < a; i++) {
            if (singletons[i]) {
                int best = -1;

                x = i % w;
                y = i / w;

                if (x > 0 && dsf_size(dsf, i - 1) < maxblk &&
                    (best == -1 || revorder[i - 1] < revorder[best]))
                    best = i - 1;

                if (x + 1 < w && dsf_size(dsf, i + 1) < maxblk &&
                    (best == -1 || revorder[i + 1] < revorder[best]))
                    best = i + 1;

                if (y > 0 && dsf_size(dsf, i - w) < maxblk &&
                    (best == -1 || revorder[i - w] < revorder[best]))
                    best = i - w;

                if (y + 1 < w && dsf_size(dsf, i + w) < maxblk &&
                    (best == -1 || revorder[i + w] < revorder[best]))
                    best = i + w;

                if (best >= 0) {
                    singletons[i] = singletons[best] = FALSE;
                    dsf_merge(dsf, i, best);
                }
            }
        }

        /* Quit and start again if we have any singletons left over */
        for (i = 0; i < a; i++)
            if (singletons[i])
                break;

        if (i < a)
            continue;

        /*
         * Decide acceptable clues for each block.
         */
#define F_ADD 0x001
#define F_SUB 0x002
#define F_MUL 0x004
#define F_DIV 0x008
#define F_EXP 0x010  /* Exponentiation: base^exp */
#define F_MOD 0x020  /* Modulo: larger % smaller */
#define F_GCD 0x040  /* Greatest common divisor */
#define F_LCM 0x080  /* Least common multiple */
#define BAD_SHIFT 8  /* Increased for 8 clue types */

        for (i = 0; i < a; i++) {
            singletons[i] = 0;
            j = dsf_canonify(dsf, i);
            k = dsf_size(dsf, j);
            if (params->multiplication_only)
                singletons[j] = F_MUL;
            else if (j == i && k > 2) {
                singletons[j] |= F_ADD | F_MUL;
            } else if (j != i && k == 2) {
                int p = grid[j], q = grid[i], v;
                if (p < q) {
                    int t = p;
                    p = q;
                    q = t;
                }

                v = p + q;
                if (v > 4 && v < 2 * w - 2)
                    singletons[j] |= F_ADD;
                else
                    singletons[j] |= F_ADD << BAD_SHIFT;

                v = p * q;
                n = 0;
                for (k = 1; k <= w; k++)
                    if (v % k == 0 && v / k <= w && v / k != k)
                        n++;
                if (n <= 2 && diff > DIFF_NORMAL)
                    singletons[j] |= F_MUL << BAD_SHIFT;
                else
                    singletons[j] |= F_MUL;

                v = p - q;
                if (v < w - 1)
                    singletons[j] |= F_SUB;

                /* Division: disabled in Zero-Inclusive, Negative, and Modular modes */
                if (!HAS_MODE(params->mode_flags, MODE_ZERO_INCLUSIVE) &&
                    !HAS_MODE(params->mode_flags, MODE_NEGATIVE) &&
                    !HAS_MODE(params->mode_flags, MODE_MODULAR) &&
                    p % q == 0 && 2 * (p / q) <= w)
                    singletons[j] |= F_DIV;

                /* Exponentiation: only when mode is enabled */
                if (HAS_MODE(params->mode_flags, MODE_EXPONENT) && q >= 2) {
                    long exp_val = 1;
                    int valid = 1;
                    int e;
                    for (e = 0; e < q && valid; e++) {
                        exp_val *= p;
                        if (exp_val > 10000) valid = 0;
                    }
                    if (valid && exp_val <= 10000)
                        singletons[j] |= F_EXP;
                }

                /* Number Theory operations: only when mode is enabled */
                if (HAS_MODE(params->mode_flags, MODE_NUMBER_THEORY)) {
                    /* Modulo: larger % smaller, only if result is non-zero */
                    v = p % q;
                    if (v > 0 && v < w)
                        singletons[j] |= F_MOD;

                    /* GCD: greatest common divisor */
                    v = (int)gcd_helper(p, q);
                    if (v >= 1 && v <= w)
                        singletons[j] |= F_GCD;

                    /* LCM: least common multiple (if not too large) */
                    v = (int)lcm_helper(p, q);
                    if (v > 0 && v <= 100)  /* LCM can grow large */
                        singletons[j] |= F_LCM;
                }
            }
        }

        /* Choose clues */
        shuffle(order, a, sizeof(*order), rs);
        for (i = 0; i < a; i++)
            clues[i] = 0;
        while (1) {
            int done_something = FALSE;

            for (k = 0; k < 8; k++) {
                long clue;
                int good, bad;
                switch (k) {
                case 0:
                    clue = C_DIV;
                    good = F_DIV;
                    break;
                case 1:
                    clue = C_SUB;
                    good = F_SUB;
                    break;
                case 2:
                    clue = C_MUL;
                    good = F_MUL;
                    break;
                case 3:
                    clue = C_ADD;
                    good = F_ADD;
                    break;
                case 4:
                    clue = C_EXP;
                    good = F_EXP;
                    break;
                case 5:
                    clue = C_MOD;
                    good = F_MOD;
                    break;
                case 6:
                    clue = C_GCD;
                    good = F_GCD;
                    break;
                case 7:
                    clue = C_LCM;
                    good = F_LCM;
                    break;
                default:
                    continue;  /* Safety fallback */
                }

                for (i = 0; i < a; i++) {
                    j = order[i];
                    if (singletons[j] & good) {
                        clues[j] = clue;
                        singletons[j] = 0;
                        break;
                    }
                }
                if (i == a) {
                    bad = good << BAD_SHIFT;
                    for (i = 0; i < a; i++) {
                        j = order[i];
                        if (singletons[j] & bad) {
                            clues[j] = clue;
                            singletons[j] = 0;
                            break;
                        }
                    }
                }
                if (i < a)
                    done_something = TRUE;
            }

            if (!done_something)
                break;
        }
#undef F_ADD
#undef F_SUB
#undef F_MUL
#undef F_DIV
#undef F_EXP
#undef F_MOD
#undef F_GCD
#undef F_LCM
#undef BAD_SHIFT

        /* Calculate clue values */
        for (i = 0; i < a; i++) {
            j = dsf_canonify(dsf, i);
            if (j == i) {
                cluevals[j] = grid[i];
            } else {
                switch (clues[j]) {
                case C_ADD:
                    cluevals[j] += grid[i];
                    break;
                case C_MUL:
                    cluevals[j] *= grid[i];
                    break;
                case C_SUB:
                    cluevals[j] = labs(cluevals[j] - grid[i]);
                    break;
                case C_DIV: {
                    int d1 = cluevals[j], d2 = grid[i];
                    if (d1 == 0 || d2 == 0)
                        cluevals[j] = 0;
                    else
                        cluevals[j] = d2 / d1 + d1 / d2;
                } break;
                case C_EXP: {
                    /* Exponentiation: smaller^larger (base^exp) */
                    long base, exp_val;
                    long result = 1;
                    int e;
                    if (cluevals[j] <= grid[i]) {
                        base = cluevals[j];
                        exp_val = grid[i];
                    } else {
                        base = grid[i];
                        exp_val = cluevals[j];
                    }
                    for (e = 0; e < exp_val; e++)
                        result *= base;
                    cluevals[j] = result;
                } break;
                case C_MOD: {
                    /* Modulo: larger % smaller (2-cell only) */
                    long larger = cluevals[j] > grid[i] ? cluevals[j] : grid[i];
                    long smaller = cluevals[j] <= grid[i] ? cluevals[j] : grid[i];
                    cluevals[j] = smaller > 0 ? larger % smaller : 0;
                } break;
                case C_GCD:
                    /* GCD: accumulate gcd(current, next) */
                    cluevals[j] = gcd_helper(cluevals[j], grid[i]);
                    break;
                case C_LCM:
                    /* LCM: accumulate lcm(current, next) */
                    cluevals[j] = lcm_helper(cluevals[j], grid[i]);
                    break;
                }
            }
        }

        /* Modular mode: apply mod N to all clue values, keeping in range 1..N */
        if (HAS_MODE(params->mode_flags, MODE_MODULAR)) {
            for (i = 0; i < a; i++) {
                j = dsf_canonify(dsf, i);
                if (j == i) {
                    cluevals[j] = ((cluevals[j] - 1) % w + w) % w + 1;
                }
            }
        }

        for (i = 0; i < a; i++) {
            j = dsf_canonify(dsf, i);
            if (j == i) {
                clues[j] |= cluevals[j];
            }
        }

        /*
         * See if the game can be solved at the specified difficulty.
         */
        if (diff > 0) {
            memset(soln, 0, a * sizeof(digit));
            ret = solver(w, dsf, clues, soln, diff - 1, params->mode_flags);
            if (ret <= diff - 1)
                continue;
        }
        memset(soln, 0, a * sizeof(digit));
        ret = solver(w, dsf, clues, soln, diff, params->mode_flags);
        if (ret != diff)
            continue;

        /* Success! */
        break;
    }

    if (attempts >= max_retries) {
        /* Failed to generate a puzzle from this grid within retry limit */
        sfree(grid);
        sfree(order);
        sfree(revorder);
        sfree(singletons);
        sfree(dsf);
        sfree(clues);
        sfree(cluevals);
        sfree(soln);
        return NULL;
    }

    /* Encode description */
    desc = snewn(40 * a, char);
    p = desc;

    for (i = 0; i < a; i++) {
        j = dsf_canonify(dsf, i);
        p += sprintf(p, "%02d", j);
        if (i < a - 1)
            *p++ = ',';
    }

    *p++ = ';';

    for (i = 0; i < a; i++) {
        j = dsf_canonify(dsf, i);
        if (j == i) {
            switch (clues[j] & CMASK) {
            case C_ADD:
                *p++ = 'a';
                break;
            case C_SUB:
                *p++ = 's';
                break;
            case C_MUL:
                *p++ = 'm';
                break;
            case C_DIV:
                *p++ = 'd';
                break;
            case C_EXP:
                *p++ = 'e';
                break;
            case C_MOD:
                *p++ = 'o';  /* 'o' for m'o'dulo */
                break;
            case C_GCD:
                *p++ = 'g';
                break;
            case C_LCM:
                *p++ = 'l';
                break;
            }
            p += sprintf(p, "%05ld", clues[j] & ~CMASK);
            *p++ = ',';
        }
    }

    --p;
    *p++ = '\0';

    desc = sresize(desc, p - desc, char);

    /* Encode solution */
    assert(memcmp(soln, grid, a * sizeof(digit)) == 0);
    *aux = snewn(a + 2, char);

    for (i = 0; i < a; i++)
        (*aux)[i] = '0' + soln[i];
    (*aux)[a] = 'e';
    (*aux)[a + 1] = '\0';

    sfree(grid);
    sfree(order);
    sfree(revorder);
    sfree(singletons);
    sfree(dsf);
    sfree(clues);
    sfree(cluevals);
    sfree(soln);

    return desc;
}


/* ======================================================================
 * LEGACY DESKTOP GUI CODE REMOVED
 * ======================================================================
 *
 * The following functionality from Simon Tatham's original desktop
 * implementation has been removed as Orthogon uses Jetpack Compose
 * for all rendering (GameScreen.kt, VictoryAnimation.kt).
 *
 * Removed sections (~1280 lines):
 *   - Gameplay: validate_desc, new_game, dup_game, free_game, solve_game
 *   - UI state: new_ui, free_ui, encode_ui, decode_ui, game_changed_state
 *   - Validation: check_errors (duplicate detection, Latin square rules)
 *   - Move handling: interpret_move, execute_move
 *   - Desktop drawing: game_compute_size, game_set_size, game_colours,
 *     game_new_drawstate, game_free_drawstate, draw_tile, game_redraw
 *   - Animation: game_anim_length, game_flash_length
 *   - Printing: game_print_size, game_print
 *   - Standalone solver: main() with command-line interface
 *
 * TODO(android-tv): If supporting Android TV/Chromebook with D-pad
 *   navigation, consider reimplementing cursor-based input handling
 *   from interpret_move() in Compose with focusable cells.
 *
 * TODO(validation): The check_errors() logic could be extracted to a
 *   separate validation module for real-time error highlighting in UI.
 *   Current implementation handles this in GameViewModel.kt.
 *
 * TODO(hints): The solve_game() function could provide step-by-step
 *   hints if integrated with the solver. Consider exposing solver
 *   state through JNI for "show next logical step" feature.
 *
 * Reference: Original code preserved in git history (commit before
 * this cleanup) and at https://www.chiark.greenend.org.uk/~sgtatham/puzzles/
 */

/* vim: set shiftwidth=4 tabstop=8: */
