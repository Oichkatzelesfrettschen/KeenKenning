#include <assert.h>
#include <string.h>

#include "keen_solver.h"

/* ----------------------------------------------------------------------
 * Solver.
 */

struct solver_ctx {
    int w, diff;
    int nboxes;
    int *boxes, *boxlist, *whichbox;
    clue_t* clues;
    digit *soln;
    digit *dscratch;
    int *iscratch;
    int mode_flags; /* Mode flags for Killer, Modular, etc. */
};

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
    return (a / g) * b; /* Avoid overflow by dividing first */
}

static long gcd_array(digit* arr, int n) {
    if (n == 0) return 0;
    long result = arr[0];
    for (int i = 1; i < n; i++) {
        result = gcd_helper(result, arr[i]);
        if (result == 1) return 1; /* Early exit: GCD can't go lower */
    }
    return result;
}

static long lcm_array(digit* arr, int n) {
    if (n == 0) return 0;
    long result = arr[0];
    for (int i = 1; i < n; i++) {
        result = lcm_helper(result, arr[i]);
        if (result > 1000000) return result; /* Prevent overflow */
    }
    return result;
}

static void solver_clue_candidate(struct solver_ctx* ctx, int diff, int box) {
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
            if (seen & bit) return; /* Duplicate found, reject this candidate */
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
        for (j = 0; j < n; j++) mask |= 1 << ctx->dscratch[j];
        for (j = 0; j < n; j++) ctx->iscratch[j] |= mask;
    } else if (diff == DIFF_NORMAL) {
        /*
         * Normal-mode deductions: we process the information in
         * dscratch in the obvious way.
         */
        for (j = 0; j < n; j++) ctx->iscratch[j] |= 1 << ctx->dscratch[j];
    } else if (diff == DIFF_HARD) {
        /*
         * Hard-mode deductions: instead of ruling things out
         * _inside_ the clue box, we look for numbers which occur in
         * a given row or column in all candidate layouts, and rule
         * them out of all squares in that row or column that
         * _aren't_ part of this clue box.
         */
        int* sq = ctx->boxlist + ctx->boxes[box];

        for (j = 0; j < 2 * w; j++) ctx->iscratch[2 * w + j] = 0;
        for (j = 0; j < n; j++) {
            int x = sq[j] / w, y = sq[j] % w;
            ctx->iscratch[2 * w + x] |= 1 << ctx->dscratch[j];
            ctx->iscratch[3 * w + y] |= 1 << ctx->dscratch[j];
        }
        for (j = 0; j < 2 * w; j++) ctx->iscratch[j] &= ctx->iscratch[2 * w + j];
    }
}

static int solver_common(struct latin_solver* solver, void* vctx, int diff) {
    struct solver_ctx* ctx = (struct solver_ctx*)vctx;
    int w = ctx->w;
    int box, i, j, k;
    int ret = 0, total;

    /*
     * Iterate over each clue box and deduce what we can.
     */
    for (box = 0; box < ctx->nboxes; box++) {
        int* sq = ctx->boxlist + ctx->boxes[box];
        int n = ctx->boxes[box + 1] - ctx->boxes[box];
        unsigned long uclue = (unsigned long)ctx->clues[box];
        unsigned long value = uclue & ~CMASK;
        unsigned long op = uclue & CMASK;

        /*
         * Initialise ctx->iscratch for this clue box. At different
         * difficulty levels we must initialise a different amount of
         * it to different things; see the comments in
         * solver_clue_candidate explaining what each version does.
         */
        if (diff == DIFF_HARD) {
            for (i = 0; i < 2 * w; i++) ctx->iscratch[i] = (1 << (w + 1)) - (1 << 1);
        } else {
            for (i = 0; i < n; i++) ctx->iscratch[i] = 0;
        }

                switch ((unsigned long)op) {

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

                            j = (int)((op == C_SUB ? (long)((unsigned long)i + (unsigned long)value) : (long)((unsigned long)i * (unsigned long)value)));

                            if (j > w) break;

        

                            /* (i,j) is a valid digit pair. Try it both ways round. */

        

                            if (solver->cube[sq[0] * w + i - 1] && solver->cube[sq[1] * w + j - 1]) {

                                ctx->dscratch[0] = (digit)i;

                                ctx->dscratch[1] = (digit)j;

                                solver_clue_candidate(ctx, diff, box);

                            }

        

                            if (solver->cube[sq[0] * w + j - 1] && solver->cube[sq[1] * w + i - 1]) {

                                ctx->dscratch[0] = (digit)j;

                                ctx->dscratch[1] = (digit)i;

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

                                unsigned long result = 1;

                                int e;

                                for (e = 0; e < j; e++) {

                                    result *= (unsigned long)i;

                                    if (result > value) break; /* Early exit for overflow */

                                }

                                if (result != value) continue;

        

                                /* (i,j) is base^exp = value. Try both cell orderings. */

                                if (solver->cube[sq[0] * w + i - 1] && solver->cube[sq[1] * w + j - 1]) {

                                    ctx->dscratch[0] = (digit)i;

                                    ctx->dscratch[1] = (digit)j;

                                    solver_clue_candidate(ctx, diff, box);

                                }

        

                                if (solver->cube[sq[0] * w + j - 1] && solver->cube[sq[1] * w + i - 1]) {

                                    ctx->dscratch[0] = (digit)j;

                                    ctx->dscratch[1] = (digit)i;

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

                        total = (int)value; /* start with the identity */

                        while (1) {

                            if (i < n) {

                                /*

                                 * Find the next valid value for cell i.

                                 */

                                for (j = ctx->dscratch[i] + 1; j <= w; j++) {

                                    if (op == C_ADD ? (total < j) : ((long)((unsigned long)total % (unsigned long)value) != 0))

                                        continue; /* this one won't fit */

                                    if (!solver->cube[sq[i] * w + j - 1])

                                        continue; /* this one is ruled out already */

                                    for (k = 0; k < i; k++)

                                        if (ctx->dscratch[k] == j &&

                                            (sq[k] % w == sq[i] % w || sq[k] / w == sq[i] / w))

                                            break; /* clashes with another row/col */

                                    if (k < i) continue;

        

                                    /* Found one. */

                                    break;

                                }

        

                                if (j > w) {

                                    /* No valid values left; drop back. */

                                    i--;

                                    if (i < 0) break; /* overall iteration is finished */

                                    if (op == C_ADD)

                                        total += ctx->dscratch[i];

                                    else

                                        total *= ctx->dscratch[i];

                                } else {

                                    /* Got a valid value; store it and move on. */

                                    ctx->dscratch[i++] = (digit)j;

                                    if (op == C_ADD)

                                        total -= j;

                                    else

                                        total /= j;

                                    ctx->dscratch[i] = 0;

                                }

                            } else {

                                if (total == (op == C_ADD ? 0 : 1)) solver_clue_candidate(ctx, diff, box);

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

        

                        for (i = 1; i <= w; i++) {               /* divisor */

                            if (value >= (unsigned long)i) continue;  /* remainder must be < divisor */

                            for (j = 1; j <= w; j++) {           /* dividend */

                                if (j <= i) continue;            /* dividend > divisor for meaningful mod */

                                if ((unsigned long)(j % i) != value) continue;

        

                                /* (j, i) is a valid pair: j % i = value */

                                if (solver->cube[sq[0] * w + j - 1] && solver->cube[sq[1] * w + i - 1]) {

                                    ctx->dscratch[0] = (digit)j;

                                    ctx->dscratch[1] = (digit)i;

                                    solver_clue_candidate(ctx, diff, box);

                                }

                                if (solver->cube[sq[0] * w + i - 1] && solver->cube[sq[1] * w + j - 1]) {

                                    ctx->dscratch[0] = (digit)i;

                                    ctx->dscratch[1] = (digit)j;

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

                                    if ((long)((unsigned long)j % (unsigned long)value) != 0) continue; /* Must be multiple of GCD */

                                    if (!solver->cube[sq[i] * w + j - 1]) continue;

                                    /* Check Latin square constraint */

                                    for (k = 0; k < i; k++)

                                        if (ctx->dscratch[k] == j &&

                                            (sq[k] % w == sq[i] % w || sq[k] / w == sq[i] / w))

                                            break;

                                    if (k < i) continue;

                                    break;

                                }

        

                                if (j > w) {

                                    i--;

                                    if (i < 0) break;

                                } else {

                                    ctx->dscratch[i++] = (digit)j;

                                    ctx->dscratch[i] = 0;

                                }

                            } else {

                                /* Check if GCD of collected digits equals value */

                                if ((unsigned long)gcd_array(ctx->dscratch, n) == value)

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

                                    if ((long)((unsigned long)value % (unsigned long)j) != 0) continue; /* Must divide LCM */

                                    if (!solver->cube[sq[i] * w + j - 1]) continue;

                                    /* Check Latin square constraint */

                                    for (k = 0; k < i; k++)

                                        if (ctx->dscratch[k] == j &&

                                            (sq[k] % w == sq[i] % w || sq[k] / w == sq[i] / w))

                                            break;

                                    if (k < i) continue;

                                    break;

                                }

        

                                if (j > w) {

                                    i--;

                                    if (i < 0) break;

                                } else {

                                    ctx->dscratch[i++] = (digit)j;

                                    ctx->dscratch[i] = 0;

                                }

                            } else {

                                /* Check if LCM of collected digits equals value */

                                if ((unsigned long)lcm_array(ctx->dscratch, n) == value)

                                    solver_clue_candidate(ctx, diff, box);

                                i--;

                            }

                        }

                        break;

        

                    case C_XOR:

                        /*

                         * XOR cages: XOR of all digits = clue value.

                         * Bitwise exclusive-or is associative and commutative.

                         * XOR has HIGH AMBIGUITY: many digit combinations produce same result,

                         * making it excellent for increasing puzzle difficulty.

                         *

                         * Properties:

                         *   - XOR(a) = a

                         *   - XOR(a,b) = a ^ b

                         *   - XOR(a,a) = 0  (self-inverse)

                         *   - XOR(a,0) = a  (identity)

                         */

                        i = 0;

                        ctx->dscratch[i] = 0;

                        total = 0; /* XOR identity is 0 */

                        while (1) {

                            if (i < n) {

                                /* Find next valid digit */

                                for (j = ctx->dscratch[i] + 1; j <= w; j++) {

                                    if (!solver->cube[sq[i] * w + j - 1]) continue;

                                    /* Check Latin square constraint */

                                    for (k = 0; k < i; k++)

                                        if (ctx->dscratch[k] == j &&

                                            (sq[k] % w == sq[i] % w || sq[k] / w == sq[i] / w))

                                            break;

                                    if (k < i) continue;

                                    break;

                                }

        

                                if (j > w) {

                                    i--;

                                    if (i < 0) break;

                                    total ^= (int)ctx->dscratch[i]; /* Undo XOR (self-inverse) */

                                } else {

                                    ctx->dscratch[i++] = (digit)j;

                                    total ^= j; /* Apply XOR */

                                    ctx->dscratch[i] = 0;

                                }

                            } else {

                                /* Check if XOR of collected digits equals value */

                                if (total == (int)value) solver_clue_candidate(ctx, diff, box);

                                i--;

                                total ^= (int)ctx->dscratch[i]; /* Undo XOR */

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
            if (ret) return ret;
        }
    }

    return ret;
}

static int solver_easy(struct latin_solver* solver, void* vctx) {
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
    struct solver_ctx* ctx = (struct solver_ctx*)vctx;
    if (ctx->diff > DIFF_EASY) return 0;
    return solver_common(solver, vctx, DIFF_EASY);
}

static int solver_normal(struct latin_solver* solver, void* vctx) {
    return solver_common(solver, vctx, DIFF_NORMAL);
}

static int solver_hard(struct latin_solver* solver, void* vctx) {
    return solver_common(solver, vctx, DIFF_HARD);
}

/*
 * EXTREME: Uses all Keen-specific deductions at HARD level.
 * The additional difficulty comes from latin.c's forcing chains.
 */
static int solver_extreme(struct latin_solver* solver, void* vctx) {
    return solver_common(solver, vctx, DIFF_HARD);
}

/*
 * UNREASONABLE: Uses all Keen-specific deductions at HARD level.
 * The additional difficulty comes from latin.c's limited recursion.
 * Puzzles at this level may require some trial-and-error.
 */
static int solver_unreasonable(struct latin_solver* solver, void* vctx) {
    return solver_common(solver, vctx, DIFF_HARD);
}

/*
 * LUDICROUS: Maximum difficulty level (beyond upstream).
 * Uses all available techniques including full backtracking.
 * Puzzles may require extensive trial-and-error to solve.
 */
static int solver_ludicrous(struct latin_solver* solver, void* vctx) {
    return solver_common(solver, vctx, DIFF_HARD);
}

/*
 * INCOMPREHENSIBLE: Ultimate difficulty level.
 * Puzzles at this level may require exceptionally deep recursion,
 * multiple simultaneous hypotheses, and advanced constraint propagation.
 * Only for the most dedicated puzzle solvers.
 */
static int solver_incomprehensible(struct latin_solver* solver, void* vctx) {
    return solver_common(solver, vctx, DIFF_HARD);
}

#define SOLVER(upper, title, func, lower) func,
static usersolver_t const keen_solvers[] = {DIFFLIST(SOLVER)};

int keen_solver(int w, int* dsf, clue_t* clues, digit* soln, int maxdiff, int mode_flags) {
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
        if (dsf_canonify(dsf, i) == i) ctx.nboxes++;
    ctx.boxlist = snewn((size_t)a, int);
    ctx.boxes = snewn((size_t)(ctx.nboxes + 1), int);
    ctx.clues = snewn((size_t)ctx.nboxes, clue_t);
    ctx.whichbox = snewn((size_t)a, int);
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

    ctx.dscratch = snewn((size_t)(a + 1), digit);
    ctx.iscratch = snewn((size_t)max(a + 1, 4 * w), int);

    /*
     * latin_solver difficulty mapping for 7-level system:
     *   diff_simple    = DIFF_EASY           (basic single-candidate deductions)
     *   diff_set_0     = DIFF_NORMAL         (pointing pairs, box/line reduction)
     *   diff_set_1     = DIFF_HARD           (naked/hidden sets, X-wing)
     *   diff_forcing   = DIFF_EXTREME        (forcing chains, region analysis)
     *   diff_recursive = DIFF_INCOMPREHENSIBLE (deep recursion, full backtracking)
     *
     * UNREASONABLE and LUDICROUS are intermediate levels between EXTREME
     * and INCOMPREHENSIBLE, requiring progressively more trial-and-error.
     */
    ret = latin_solver(soln, w, maxdiff, DIFF_EASY, DIFF_NORMAL, DIFF_HARD, DIFF_EXTREME,
                       DIFF_INCOMPREHENSIBLE, keen_solvers, &ctx, nullptr, nullptr);

    sfree(ctx.dscratch);
    sfree(ctx.iscratch);
    sfree(ctx.whichbox);
    sfree(ctx.boxlist);
    sfree(ctx.boxes);
    sfree(ctx.clues);

    return ret;
}
