/*
 * keen_generate.c: KenKen puzzle generation
 *
 * Extracted from keen.c to keep generation logic isolated.
 */

#include "keen.h"
#include "keen_internal.h"
#include "keen_solver.h"

#include <ctype.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    return (a / g) * b; /* Avoid overflow by dividing first */
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
#define MAXBLK_KILLER 16
#define MAXBLK MAXBLK_STANDARD /* Legacy compatibility */

/*
 * Get effective max block size based on mode AND difficulty.
 * Larger cages create more complex constraint interactions,
 * which are more likely to require advanced solving techniques.
 *
 * Difficulty scaling:
 * - EASY/NORMAL: Standard 6-cell max (simple constraints)
 * - HARD/EXTREME: 8-cell max (moderate complexity)
 * - UNREASONABLE+: 10-cell max (complex interactions)
 * - KILLER mode: Always 16-cell (Killer Sudoku conventions)
 */
static inline int get_maxblk_for_diff(int mode_flags, int diff) {
    if (HAS_MODE(mode_flags, MODE_KILLER)) {
        return MAXBLK_KILLER;
    }
    if (diff >= DIFF_UNREASONABLE) {
        return 10; /* Large cages for very hard puzzles */
    }
    if (diff >= DIFF_HARD) {
        return 8; /* Medium-large cages for hard puzzles */
    }
    return MAXBLK_STANDARD;
}

/* Legacy wrapper for mode-only queries */
[[maybe_unused]] static inline int get_maxblk(int mode_flags) {
    return get_maxblk_for_diff(mode_flags, DIFF_NORMAL);
}

/*
 * Get minimum cage size for a mode.
 * Killer mode typically uses larger minimum cages (2-4 cells).
 */
static inline int get_minblk(int mode_flags) {
    if (HAS_MODE(mode_flags, MODE_KILLER)) {
        return 2; /* Killer mode: minimum 2 cells per cage */
    }
    return 1; /* Standard: allow singletons */
}

/* ----------------------------------------------------------------------
 * Grid generation.
 */

[[maybe_unused]] static char* encode_block_structure(char* p, int w, int* dsf) {
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
            edge = true; /* terminating virtual edge */
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
            while (currrun > 25) *p++ = 'z', currrun -= 25;
            if (currrun)
                *p++ = (char)('a' - 1 + currrun);
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

        for (i = 0; r + i < p && r[i] == c; i++);
        r += i;

        if (i == 2) {
            *q++ = c;
        } else if (i > 2) {
            q += sprintf(q, "%d", i);
        }
    }

    return q;
}

[[maybe_unused]] static char* parse_block_structure(const char** p, int w, int* dsf) {
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
                while (**p && isdigit((unsigned char)**p)) (*p)++;
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
            if (pos >= 2 * w * (w - 1)) return "Too much data in block structure specification";
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
            if (pos > 2 * w * (w - 1) + 1) return "Too much data in block structure specification";
        }
    }

    /*
     * When desc is exhausted, we expect to have gone exactly
     * one space _past_ the end of the grid, due to the dummy
     * edge at the end.
     */
    if (pos != 2 * w * (w - 1) + 1) return "Not enough data in block structure specification";

    return nullptr;
}

/*
 * Cage Mutation for Difficulty Elevation
 *
 * When a puzzle is too easy, we try to increase difficulty by merging
 * adjacent cages. Larger cages create more complex constraint interactions,
 * which often require more advanced solving techniques.
 *
 * Strategy:
 * 1. Find pairs of adjacent cages where combined size <= maxblk
 * 2. Merge them (union in DSF)
 * 3. Recalculate the merged cage's clue
 * 4. Re-verify difficulty
 *
 * This is more efficient than regenerating entirely because we preserve
 * the Latin square and only modify cage structure.
 */

/*
 * Find and merge one pair of adjacent cages to increase difficulty.
 * Returns true if a merge was performed, false if no valid merge exists.
 *
 * Parameters:
 *   w        - grid width
 *   dsf      - disjoint set forest representing cage structure
 *   grid     - the Latin square solution
 *   clues    - clue array (operation | value)
 *   cluevals - working array for clue values
 *   maxblk   - maximum allowed cage size
 *   rs       - random state for selecting which pair to merge
 */
static int try_merge_cages(int w, int* dsf, digit* grid, clue_t* clues,
                           [[maybe_unused]] clue_t* cluevals, int maxblk, random_state* rs) {
    int a = w * w;
    int i, x, y, ni;
    int *candidates, ncand = 0;

    /*
     * Build list of candidate merge pairs (cell, neighbor) where:
     * - Cell and neighbor are in different cages
     * - Combined cage size <= maxblk
     * - We store as (cell * a + neighbor) to encode both indices
     */
    candidates = snewn((size_t)(a * 4), int); /* Max 4 neighbors per cell */

    for (i = 0; i < a; i++) {
        int ci = dsf_canonify(dsf, i);
        int si = dsf_size(dsf, i);
        x = i % w;
        y = i / w;

        /* Check right neighbor */
        if (x + 1 < w) {
            ni = i + 1;
            int cni = dsf_canonify(dsf, ni);
            if (ci != cni && si + dsf_size(dsf, ni) <= maxblk) {
                candidates[ncand++] = i * a + ni;
            }
        }

        /* Check bottom neighbor */
        if (y + 1 < w) {
            ni = i + w;
            int cni = dsf_canonify(dsf, ni);
            if (ci != cni && si + dsf_size(dsf, ni) <= maxblk) {
                candidates[ncand++] = i * a + ni;
            }
        }
    }

    if (ncand == 0) {
        sfree(candidates);
        return false; /* No valid merge candidates */
    }

    /*
     * Randomly select a merge pair to avoid bias.
     */
    int choice = (int)random_upto(rs, (unsigned long)ncand);
    int c1 = candidates[choice] / a;
    int c2 = candidates[choice] % a;
    sfree(candidates);

    /*
     * Get canonical representatives before merge.
     */
    int canon1 = dsf_canonify(dsf, c1);
    int canon2 = dsf_canonify(dsf, c2);

    /*
     * Perform the merge in DSF.
     */
    dsf_merge(dsf, c1, c2);
    int new_canon = dsf_canonify(dsf, c1);

    /*
     * Calculate new clue value by iterating over merged cage cells.
     * Use ADD for merged cages (most constraining for difficulty).
     */
    long new_val = 0;
    for (i = 0; i < a; i++) {
        if (dsf_canonify(dsf, i) == new_canon) {
            new_val += grid[i];
        }
    }

    /*
     * Update clues array - clear old entries, set new one.
     * C_ADD = 0x10000000L (defined at top of file)
     */
    clues[canon1] = 0;
    clues[canon2] = 0;
    clues[new_canon] = C_ADD | (unsigned long)new_val;

    return true;
}

/*
 * Recalculate all clues after cage mutations.
 * Call this after modifying DSF to ensure clue consistency.
 */
[[maybe_unused]] static void recalculate_clues(int w, int* dsf, digit* grid, clue_t* clues,
                              long*  cluevals,
                              [[maybe_unused]] int mode_flags,
                              [[maybe_unused]] random_state* rs) {
    int a = w * w;
    int i, j;

    /* Reset clues and cluevals */
    for (i = 0; i < a; i++) {
        clues[i] = 0;
        cluevals[i] = 0;
    }

    /*
     * For each cage (identified by canonical element), calculate clue.
     * We use the same logic as the main generation loop.
     */
    for (i = 0; i < a; i++) {
        j = dsf_canonify(dsf, i);
        if (i == j) {
            /* This is a cage root - determine operation */
            int n = dsf_size(dsf, i);

            if (n == 1) {
                /* Singleton - no operation, just the value */
                clues[i] = grid[i];
            } else {
                /*
                 * Multi-cell cage - calculate with ADD for difficulty.
                 * ADD creates the most constraint interactions for large cages.
                 */
                long sum = 0;
                int k;
                for (k = 0; k < a; k++) {
                    if (dsf_canonify(dsf, k) == i) {
                        sum += grid[k];
                    }
                }
                clues[i] = C_ADD | (unsigned long)sum;
            }
        }
    }
}

char* new_game_desc(const game_params* params, random_state* rs, char** aux, int interactive) {
    (void)interactive;
    int w = params->w, a = w * w;
    digit *grid, *soln;
    int *order, *revorder, *singletons, *dsf;
    clue_t* clues, *cluevals;
    int i, j, k, n, x, y, ret;
    int diff = params->diff;
    int maxblk = get_maxblk_for_diff(params->mode_flags, diff);
    (void)get_minblk(params->mode_flags); /* Reserved for future constraint validation */
    char *desc, *p;

    /*
     * If the requested difficulty cannot be achieved within max_retries,
     * return nullptr. No fallback - the user expects the exact difficulty
     * they selected. The UI will display an appropriate error message.
     */
    grid = nullptr;

    order = snewn(a, int);
    revorder = snewn(a, int);
    singletons = snewn(a, int);
    dsf = snew_dsf(a);
    clues = snewn(a, clue_t);
    cluevals = snewn(a, clue_t);
    soln = snewn(a, digit);

    /*
     * Limit retries to prevent infinite loops when mode constraints or
     * large grid sizes make valid puzzles rare. Scale with BOTH grid size
     * AND difficulty level - higher difficulties need exponentially more
     * attempts since advanced-technique-requiring puzzles are rare.
     *
     * Scaling rationale:
     * - EASY/NORMAL: Base attempts (most puzzles satisfy these)
     * - HARD: 2x multiplier (naked/hidden sets less common)
     * - EXTREME: 4x multiplier (forcing chains rare)
     * - UNREASONABLE+: 8x multiplier (very rare puzzle structures)
     */
    int difficulty_multiplier = 1;
    if (diff >= DIFF_UNREASONABLE)
        difficulty_multiplier = 8;
    else if (diff >= DIFF_EXTREME)
        difficulty_multiplier = 4;
    else if (diff >= DIFF_HARD)
        difficulty_multiplier = 2;

    int max_retries = (1000 + (a * 20)) * difficulty_multiplier;
    int attempts = 0;

    while (attempts < max_retries) {
        attempts++;
        /*
         * First construct a latin square to be the solution.
         */
        sfree(grid);
        grid = latin_generate(w, rs);

        /*
         * NOTE: Zero-Inclusive and Negative modes apply DISPLAY transformations
         * only. The internal grid stays as 1..N because the solver's cube array
         * is indexed by (digit - 1) and cannot handle 0 or negative values.
         *
         * The transformation is applied during solution encoding below.
         * Clue values are calculated from 1..N (consistent, solvable puzzles).
         *
         * - Zero mode: UI displays 0..N-1, internally 1..N
         * - Negative mode: UI displays -N/2..+N/2, internally 1..N
         */

        for (i = 0; i < a; i++) order[i] = i;
        shuffle(order, a, sizeof(*order), rs);
        for (i = 0; i < a; i++) revorder[order[i]] = i;

        for (i = 0; i < a; i++) singletons[i] = true;

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
                    singletons[i] = singletons[best] = false;
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
                    singletons[i] = singletons[best] = false;
                    dsf_merge(dsf, i, best);
                }
            }
        }

        /* Quit and start again if we have any singletons left over
         * which we weren't able to do anything at all with. */
        for (i = 0; i < a; i++)
            if (singletons[i]) break;

        if (i < a) continue;

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
#define F_EXP 0x010 /* Exponentiation: base^exp */
#define F_MOD 0x020 /* Modulo: larger % smaller */
#define F_GCD 0x040 /* Greatest common divisor */
#define F_LCM 0x080 /* Least common multiple */
#define F_XOR 0x100 /* Bitwise XOR (high ambiguity) */
#define BAD_SHIFT 9 /* Increased for 9+ clue types */

        for (i = 0; i < a; i++) {
            singletons[i] = 0;
            j = dsf_canonify(dsf, i);
            k = dsf_size(dsf, j);
            if (params->multiplication_only)
                singletons[j] = F_MUL;
            else if (j == i && k > 2) {
                singletons[j] |= F_ADD | F_MUL;
                /* XOR works great on N-cell cages with MODE_BITWISE */
                if (HAS_MODE(params->mode_flags, MODE_BITWISE)) singletons[j] |= F_XOR;
            } else if (j != i && k == 2) {
                /* Fetch the two numbers and sort them into order. */
                int p_val = grid[j], q = grid[i], v;
                if (p_val < q) {
                    int t = p_val;
                    p_val = q;
                    q = t;
                }

                /*
                 * Addition clues are always allowed, but we try to
                 * avoid sums of 3, 4, (2w-1) and (2w-2) if we can,
                 * because they're too easy - they only leave one
                 * option for the pair of numbers involved.
                 */
                v = p_val + q;
                if (v > 4 && v < 2 * w - 2)
                    singletons[j] |= F_ADD;
                else
                    singletons[j] |= F_ADD << BAD_SHIFT;

                /*
                 * Multiplication clues: above Normal difficulty, we
                 * prefer (but don't absolutely insist on) clues of
                 * this type which leave multiple options open.
                 */
                v = p_val * q;
                n = 0;
                for (k = 1; k <= w; k++)
                    if (v % k == 0 && v / k <= w && v / k != k) n++;
                if (n <= 2 && diff > DIFF_NORMAL)
                    singletons[j] |= F_MUL << BAD_SHIFT;
                else
                    singletons[j] |= F_MUL;

                /*
                 * Subtraction: we completely avoid a difference of
                 * w-1.
                 */
                v = p_val - q;
                if (v < w - 1) singletons[j] |= F_SUB;

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
                    !HAS_MODE(params->mode_flags, MODE_MODULAR) && p_val % q == 0 &&
                    2 * (p_val / q) <= w)
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
                        exp_val *= p_val;
                        if (exp_val > 10000) valid = 0; /* Clue too large */
                    }
                    if (valid && exp_val <= 10000) singletons[j] |= F_EXP;
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
                    v = p_val % q;
                    if (v > 0 && v < q) singletons[j] |= F_MOD;

                    /*
                     * GCD: gcd(p, q). Difficulty-aware preference:
                     * - Easy/Normal: Prefer GCD > 1 (more informative constraint)
                     * - Hard+: PREFER GCD = 1 (maximally ambiguous - any coprime pair works)
                     *
                     * GCD=1 is the most ambiguous constraint possible for 2-cell cages,
                     * forcing the solver to use advanced techniques rather than direct deduction.
                     */
                    v = (int)gcd_helper(p_val, q);
                    if (diff >= DIFF_HARD) {
                        /* Hard+: GCD=1 is GOOD (high ambiguity = harder) */
                        if (v == 1)
                            singletons[j] |= F_GCD;
                        else if (v > 1 && v < q)
                            singletons[j] |= F_GCD << BAD_SHIFT; /* Less ambiguous */
                    } else {
                        /* Easy/Normal: GCD > 1 is GOOD (more informative) */
                        if (v > 1 && v < q)
                            singletons[j] |= F_GCD;
                        else if (v == 1)
                            singletons[j] |= F_GCD << BAD_SHIFT; /* Too ambiguous */
                    }

                    /*
                     * LCM: lcm(p, q). Avoid cases where LCM > 100 (clue overflow)
                     * or LCM = p*q (coprime, trivial).
                     */
                    v = (int)lcm_helper(p_val, q);
                    if (v <= 100 && v < p_val * q)
                        singletons[j] |= F_LCM;
                    else if (v <= 100)
                        singletons[j] |= F_LCM << BAD_SHIFT;
                }

                /*
                 * XOR: p ^ q. Available when MODE_BITWISE is active.
                 * XOR has VERY HIGH ambiguity - many pairs produce the same result.
                 * Excellent for increasing puzzle difficulty.
                 *
                 * For difficulty: XOR is ALWAYS preferred at Hard+ because
                 * it provides minimal constraint information.
                 */
                if (HAS_MODE(params->mode_flags, MODE_BITWISE)) {
                    v = p_val ^ q;
                    if (diff >= DIFF_HARD)
                        singletons[j] |= F_XOR; /* Always good at hard+ */
                    else
                        singletons[j] |= F_XOR << BAD_SHIFT; /* Too ambiguous for easy */
                }
            }
        }

        /*
         * Assign clue types to blocks, balancing the distribution
         * across operation types and preferring optimal candidates.
         *
         * DIFFICULTY-AWARE OPERATION ORDERING:
         * For high difficulties, we prefer operations with HIGH AMBIGUITY
         * (many valid combinations = less constraint information), forcing
         * the solver to use advanced techniques like forcing chains.
         *
         * Ambiguity ranking (high to low):
         *   XOR > GCD (esp. GCD=1) > ADD (large cages) > LCM > MOD > SUB > MUL > DIV > EXP
         *
         * Note: O(N^2) complexity is acceptable here since N is bounded
         * by the maximum number of dominoes in a 9x9 grid.
         */
        shuffle(order, a, sizeof(*order), rs);
        for (i = 0; i < a; i++) clues[i] = 0;

        /*
         * Operation order arrays: different priorities for different difficulties.
         * Easy/Normal: Prefer constraining ops (DIV, EXP) for simpler puzzles.
         * Hard+: Prefer ambiguous ops (XOR, GCD, ADD, LCM) for harder puzzles.
         * Index mapping: 0=DIV, 1=SUB, 2=MUL, 3=ADD, 4=EXP, 5=MOD, 6=GCD, 7=LCM, 8=XOR
         */
        static const int op_order_easy[9] = {0, 1, 2, 3, 4,
                                             5, 6, 7, 8}; /* DIV...LCM,XOR (XOR last) */
        static const int op_order_hard[9] = {8, 6, 3, 7, 5,
                                             1, 2, 0, 4}; /* XOR,GCD,ADD,LCM,MOD,SUB,MUL,DIV,EXP */
        const int* op_order = (diff >= DIFF_HARD) ? op_order_hard : op_order_easy;

        while (1) {
            int done_something = false;

            for (int op_idx = 0; op_idx < 9; op_idx++) {
                long clue;
                int good, bad;
                k = op_order[op_idx]; /* Use difficulty-aware ordering */
                switch (k) {
                    case 0:
                        clue = (long)C_DIV;
                        good = F_DIV;
                        break;
                    case 1:
                        clue = (long)C_SUB;
                        good = F_SUB;
                        break;
                    case 2:
                        clue = (long)C_MUL;
                        good = F_MUL;
                        break;
                    case 3:
                        clue = (long)C_ADD;
                        good = F_ADD;
                        break;
                    case 4:
                        clue = (long)C_EXP;
                        good = F_EXP;
                        break;
                    case 5:
                        clue = (long)C_MOD;
                        good = F_MOD;
                        break;
                    case 6:
                        clue = (long)C_GCD;
                        good = F_GCD;
                        break;
                    case 7:
                        clue = (long)C_LCM;
                        good = F_LCM;
                        break;
                    case 8:
                        clue = (long)C_XOR;
                        good = F_XOR;
                        break;
                    default:
                        continue; /* Safety fallback */
                }

                for (i = 0; i < a; i++) {
                    j = order[i];
                    if (singletons[j] & good) {
                        clues[j] = (unsigned long)clue;
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
                            clues[j] = (unsigned long)clue;
                            singletons[j] = 0;
                            break;
                        }
                    }
                }
                if (i < a) done_something = true;
            }

            if (!done_something) break;
        }
#undef F_ADD
#undef F_SUB
#undef F_MUL
#undef F_DIV
#undef F_EXP
#undef F_MOD
#undef F_GCD
#undef F_LCM
#undef F_XOR
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
                        if (cluevals[j] > (clue_t)grid[i])
                            cluevals[j] = cluevals[j] - (clue_t)grid[i];
                        else
                            cluevals[j] = (clue_t)grid[i] - cluevals[j];
                        break;
                    case C_DIV: {
                        int d1 = (int)cluevals[j], d2 = grid[i];
                        if (d1 == 0 || d2 == 0)
                            cluevals[j] = 0;
                        else
                            cluevals[j] = (unsigned long)(d2 / d1 + d1 / d2); /* one is 0 :-) */
                    } break;
                    case C_EXP: {
                        /* Exponentiation: smaller^larger (base^exp) */
                        clue_t base, exp_val;
                        unsigned long result = 1;
                        int e;
                        if (cluevals[j] <= (clue_t)grid[i]) {
                            base = cluevals[j];
                            exp_val = (clue_t)grid[i];
                        } else {
                            base = (clue_t)grid[i];
                            exp_val = cluevals[j];
                        }
                        for (e = 0; e < (int)exp_val; e++) result *= (unsigned long)base;
                        cluevals[j] = result;
                    } break;
                    case C_MOD: {
                        /* Modulo: larger % smaller (2-cell only) */
                        clue_t larger = cluevals[j] > (clue_t)grid[i] ? cluevals[j] : (clue_t)grid[i];
                        clue_t smaller = cluevals[j] <= (clue_t)grid[i] ? cluevals[j] : (clue_t)grid[i];
                        cluevals[j] = smaller > 0 ? larger % smaller : 0;
                    } break;
                    case C_GCD:
                        /* GCD: accumulate gcd(current, next) */
                        cluevals[j] = (unsigned long)gcd_helper((long)cluevals[j], grid[i]);
                        break;
                    case C_LCM:
                        /* LCM: accumulate lcm(current, next) */
                        cluevals[j] = (unsigned long)lcm_helper((long)cluevals[j], grid[i]);
                        break;
                    case C_XOR:
                        /* XOR: accumulate xor(current, next) - self-inverse */
                        cluevals[j] ^= grid[i];
                        break;
                }
            }
        }

        /*
         * NOTE: Modular mode clue value wrapping is DISABLED.
         *
         * The solver's clue verification logic at lines 548-609 checks exact
         * clue values (e.g., "does this combination sum to exactly 15?").
         * Wrapping clue values with mod N breaks this verification because
         * the solver doesn't know to apply inverse modular arithmetic.
         *
         * For now, Modular mode puzzles use standard clue values but the UI
         * can indicate "mod N" in the display. Full modular arithmetic support
         * would require modifying the solver's candidate verification.
         *
         * TODO: Implement modular arithmetic in solver_common() for true mod N
         */

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
            memset(soln, 0, (size_t)a * sizeof(digit));
            ret = keen_solver(w, dsf, clues, soln, diff - 1, params->mode_flags);
            if (ret <= diff - 1) {
                /*
                 * Puzzle is too easy - try cage merging to increase difficulty.
                 * Merging adjacent cages creates more complex constraint
                 * interactions, often requiring more advanced techniques.
                 */
                int merge_attempts = 0;
                int max_merges = w * 2; /* Limit merge iterations */

                while (merge_attempts < max_merges && ret <= diff - 1) {
                    if (!try_merge_cages(w, dsf, grid, clues, cluevals, maxblk, rs)) {
                        break; /* No more merges possible */
                    }
                    merge_attempts++;

                    /* Re-test difficulty after merge */
                    memset(soln, 0, (size_t)a * sizeof(digit));
                    ret = keen_solver(w, dsf, clues, soln, diff - 1, params->mode_flags);
                }

                if (ret <= diff - 1) continue; /* Still too easy after merging - new attempt */
            }
        }
        memset(soln, 0, (size_t)a * sizeof(digit));
        ret = keen_solver(w, dsf, clues, soln, diff, params->mode_flags);
        if (ret != diff) {
            /*
             * Puzzle doesn't match target difficulty - try merging.
             * This handles cases where puzzle is harder than requested
             * (ret > diff) by trying to simplify via different cage structure.
             */
            if (ret < diff) {
                /* Too easy - merge cages to increase difficulty */
                int merge_attempts = 0;
                int max_merges = w * 2;

                while (merge_attempts < max_merges && ret < diff) {
                    if (!try_merge_cages(w, dsf, grid, clues, cluevals, maxblk, rs)) {
                        break;
                    }
                    merge_attempts++;
                    memset(soln, 0, (size_t)a * sizeof(digit));
                    ret = keen_solver(w, dsf, clues, soln, diff, params->mode_flags);
                }
            }

            if (ret != diff) continue; /* go round again */
        }

        /*
         * We've got a usable puzzle!
         */
        break;
    }

    /*
     * Check if we exhausted retries without finding a valid puzzle.
     * No fallback - user expects the exact difficulty they selected.
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
        return nullptr;
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
        if (i < a - 1) *p++ = ',';
    }

    *p++ = ';';

    for (i = 0; i < a; i++) {
        j = dsf_canonify(dsf, i);
        if (j == i) {
            switch ((unsigned long)clues[j] & CMASK) {
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
                    *p++ = 'o'; /* 'o' for m'o'dulo, 'm' taken by MUL */
                    break;
                case C_GCD:
                    *p++ = 'g';
                    break;
                case C_LCM:
                    *p++ = 'l';
                    break;
                case C_XOR:
                    *p++ = 'x'; /* XOR: x ⊕ y notation */
                    break;
            }
            p += sprintf(p, "%05" PRIu64, (uint64_t)(clues[j] & ~CMASK));

            *p++ = ',';
        }
    }

    --p;
    *p++ = '\0';

    desc = sresize(desc, p - desc, char);

    /*
     * Encode the solution in aux.
     */
    *aux = snewn(a + 2, char);
    char *auxp = *aux;
    *auxp++ = 'S'; /* Solution marker */
    for (i = 0; i < a; i++) {
        int display_val = soln[i];
        
        /* Apply mode-specific display transformations */
        if (HAS_MODE(params->mode_flags, MODE_ZERO_INCLUSIVE)) {
            display_val -= 1; /* 1..N -> 0..N-1 */
        } else if (HAS_MODE(params->mode_flags, MODE_NEGATIVE)) {
            int range = w;
            int half = range / 2;
            /* 1..N -> -half..+(range-half-1) */
            /* e.g. size 5: 1,2,3,4,5 -> -2,-1,0,1,2 */
            display_val -= (half + 1);
        }

        if (display_val >= 10)
            *auxp++ = (char)('A' + (display_val - 10));
        else if (display_val >= 0)
            *auxp++ = (char)('0' + display_val);
        else 
            *auxp++ = '?'; /* Should not happen with valid modes */
    }
    *auxp = '\0';
    *aux = sresize(*aux, auxp - *aux + 1, char);

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

char* new_game_desc_from_grid(const game_params* params, random_state* rs, digit* input_grid,
                              char** aux, int interactive) {
    (void)interactive;
    int w = params->w, a = w * w;
    digit *grid, *soln;
    int *order, *revorder, *singletons, *dsf;
    clue_t* clues, *cluevals;
    int i, j, k, n, x, y, ret;
    int diff = params->diff;
    int maxblk = get_maxblk_for_diff(params->mode_flags, diff);
    (void)get_minblk(params->mode_flags); /* Reserved for future constraint validation */
    char *desc, *p;

    /*
     * Strict difficulty enforcement for AI generation path:
     * If the requested difficulty cannot be achieved within max_retries,
     * return nullptr. No fallback - the user expects the exact difficulty
     * they selected. The UI will display an appropriate error message.
     */
    grid = nullptr;

    order = snewn(a, int);
    revorder = snewn(a, int);
    singletons = snewn(a, int);
    dsf = snew_dsf(a);
    clues = snewn(a, clue_t);
    cluevals = snewn(a, clue_t);
    soln = snewn(a, digit);

    /* Use the provided grid */
    grid = snewn(a, digit);
    memcpy(grid, input_grid, (size_t)a * sizeof(digit));

    /*
     * NOTE: Zero-Inclusive and Negative modes apply DISPLAY transformations
     * only. The internal grid stays as 1..N because the solver's cube array
     * is indexed by (digit - 1) and cannot handle 0 or negative values.
     *
     * See solution encoding below for where transformations are applied.
     */

    /*
     * Limit retries - scale with difficulty for higher success rate.
     * AI-generated grids may have specific properties that make
     * certain difficulty levels harder to achieve.
     */
    int difficulty_multiplier = 1;
    if (diff >= DIFF_UNREASONABLE)
        difficulty_multiplier = 8;
    else if (diff >= DIFF_EXTREME)
        difficulty_multiplier = 4;
    else if (diff >= DIFF_HARD)
        difficulty_multiplier = 2;

    int max_retries = 1000 * difficulty_multiplier;
    int attempts = 0;

    while (attempts < max_retries) {
        attempts++;

        /*
         * Divide the grid into arbitrarily sized blocks.
         */
        for (i = 0; i < a; i++) order[i] = i;

        shuffle(order, a, sizeof(*order), rs);

        for (i = 0; i < a; i++) revorder[order[i]] = i;

        for (i = 0; i < a; i++) singletons[i] = true;

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
                    singletons[i] = singletons[best] = false;
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
                    singletons[i] = singletons[best] = false;
                    dsf_merge(dsf, i, best);
                }
            }
        }

        /* Quit and start again if we have any singletons left over */
        for (i = 0; i < a; i++)
            if (singletons[i]) break;

        if (i < a) continue;

        /*
         * Decide acceptable clues for each block.
         */
#define F_ADD 0x001
#define F_SUB 0x002
#define F_MUL 0x004
#define F_DIV 0x008
#define F_EXP 0x010 /* Exponentiation: base^exp */
#define F_MOD 0x020 /* Modulo: larger % smaller */
#define F_GCD 0x040 /* Greatest common divisor */
#define F_LCM 0x080 /* Least common multiple */
#define F_XOR 0x100 /* Bitwise XOR (high ambiguity) */
#define BAD_SHIFT 9 /* Increased for 9+ clue types */

        for (i = 0; i < a; i++) {
            singletons[i] = 0;
            j = dsf_canonify(dsf, i);
            k = dsf_size(dsf, j);
            if (params->multiplication_only)
                singletons[j] = F_MUL;
            else if (j == i && k > 2) {
                singletons[j] |= F_ADD | F_MUL;
                /* XOR works great on N-cell cages with MODE_BITWISE */
                if (HAS_MODE(params->mode_flags, MODE_BITWISE)) singletons[j] |= F_XOR;
            } else if (j != i && k == 2) {
                int p_val = grid[j], q = grid[i], v;
                if (p_val < q) {
                    int t = p_val;
                    p_val = q;
                    q = t;
                }

                v = p_val + q;
                if (v > 4 && v < 2 * w - 2)
                    singletons[j] |= F_ADD;
                else
                    singletons[j] |= F_ADD << BAD_SHIFT;

                v = p_val * q;
                n = 0;
                for (k = 1; k <= w; k++)
                    if (v % k == 0 && v / k <= w && v / k != k) n++;
                if (n <= 2 && diff > DIFF_NORMAL)
                    singletons[j] |= F_MUL << BAD_SHIFT;
                else
                    singletons[j] |= F_MUL;

                v = p_val - q;
                if (v < w - 1) singletons[j] |= F_SUB;

                /* Division: disabled in Zero-Inclusive, Negative, and Modular modes */
                if (!HAS_MODE(params->mode_flags, MODE_ZERO_INCLUSIVE) &&
                    !HAS_MODE(params->mode_flags, MODE_NEGATIVE) &&
                    !HAS_MODE(params->mode_flags, MODE_MODULAR) && p_val % q == 0 &&
                    2 * (p_val / q) <= w)
                    singletons[j] |= F_DIV;

                /* Exponentiation: only when mode is enabled */
                if (HAS_MODE(params->mode_flags, MODE_EXPONENT) && q >= 2) {
                    long exp_val = 1;
                    int valid = 1;
                    int e;
                    for (e = 0; e < q && valid; e++) {
                        exp_val *= p_val;
                        if (exp_val > 10000) valid = 0;
                    }
                    if (valid && exp_val <= 10000) singletons[j] |= F_EXP;
                }

                /* Number Theory operations: only when mode is enabled */
                if (HAS_MODE(params->mode_flags, MODE_NUMBER_THEORY)) {
                    /* Modulo: larger % smaller, only if result is non-zero */
                    v = p_val % q;
                    if (v > 0 && v < w) singletons[j] |= F_MOD;

                    /* GCD: greatest common divisor */
                    v = (int)gcd_helper(p_val, q);
                    if (v >= 1 && v <= w) singletons[j] |= F_GCD;

                    /* LCM: least common multiple (if not too large) */
                    v = (int)lcm_helper(p_val, q);
                    if (v > 0 && v <= 100) /* LCM can grow large */
                        singletons[j] |= F_LCM;
                }

                /* XOR: available when MODE_BITWISE is active */
                if (HAS_MODE(params->mode_flags, MODE_BITWISE)) {
                    v = p_val ^ q;
                    if (diff >= DIFF_HARD)
                        singletons[j] |= F_XOR; /* Always good at hard+ */
                    else
                        singletons[j] |= F_XOR << BAD_SHIFT; /* Too ambiguous for easy */
                }
            }
        }

        /* Choose clues */
        shuffle(order, a, sizeof(*order), rs);
        for (i = 0; i < a; i++) clues[i] = 0;
        while (1) {
            int done_something = false;

            for (k = 0; k < 9; k++) {
                long clue;
                int good, bad;
                switch (k) {
                    case 0:
                        clue = (long)C_DIV;
                        good = F_DIV;
                        break;
                    case 1:
                        clue = (long)C_SUB;
                        good = F_SUB;
                        break;
                    case 2:
                        clue = (long)C_MUL;
                        good = F_MUL;
                        break;
                    case 3:
                        clue = (long)C_ADD;
                        good = F_ADD;
                        break;
                    case 4:
                        clue = (long)C_EXP;
                        good = F_EXP;
                        break;
                    case 5:
                        clue = (long)C_MOD;
                        good = F_MOD;
                        break;
                    case 6:
                        clue = (long)C_GCD;
                        good = F_GCD;
                        break;
                    case 7:
                        clue = (long)C_LCM;
                        good = F_LCM;
                        break;
                    case 8:
                        clue = (long)C_XOR;
                        good = F_XOR;
                        break;
                    default:
                        continue; /* Safety fallback */
                }

                for (i = 0; i < a; i++) {
                    j = order[i];
                    if (singletons[j] & good) {
                        clues[j] = (unsigned long)clue;
                        singletons[j] = 0;
                        break;
                    }
                }
                if (i == a) {
                    bad = good << BAD_SHIFT;
                    for (i = 0; i < a; i++) {
                        j = order[i];
                        if (singletons[j] & bad) {
                            clues[j] = (unsigned long)clue;
                            singletons[j] = 0;
                            break;
                        }
                    }
                }
                if (i < a) done_something = true;
            }

            if (!done_something) break;
        }
#undef F_ADD
#undef F_SUB
#undef F_MUL
#undef F_DIV
#undef F_EXP
#undef F_MOD
#undef F_GCD
#undef F_LCM
#undef F_XOR
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
                        if (cluevals[j] > (clue_t)grid[i])
                            cluevals[j] = cluevals[j] - (clue_t)grid[i];
                        else
                            cluevals[j] = (clue_t)grid[i] - cluevals[j];
                        break;
                    case C_DIV: {
                        int d1 = (int)cluevals[j], d2 = grid[i];
                        if (d1 == 0 || d2 == 0)
                            cluevals[j] = 0;
                        else
                            cluevals[j] = (unsigned long)(d2 / d1 + d1 / d2);
                    } break;
                    case C_EXP: {
                        /* Exponentiation: smaller^larger (base^exp) */
                        clue_t base, exp_val;
                        unsigned long result = 1;
                        int e;
                        if (cluevals[j] <= (clue_t)grid[i]) {
                            base = cluevals[j];
                            exp_val = (clue_t)grid[i];
                        } else {
                            base = (clue_t)grid[i];
                            exp_val = cluevals[j];
                        }
                        for (e = 0; e < (int)exp_val; e++) result *= (unsigned long)base;
                        cluevals[j] = result;
                    } break;
                    case C_MOD: {
                        /* Modulo: larger % smaller (2-cell only) */
                        clue_t larger = cluevals[j] > (clue_t)grid[i] ? cluevals[j] : (clue_t)grid[i];
                        clue_t smaller = cluevals[j] <= (clue_t)grid[i] ? cluevals[j] : (clue_t)grid[i];
                        cluevals[j] = smaller > 0 ? larger % smaller : 0;
                    } break;
                    case C_GCD:
                        /* GCD: accumulate gcd(current, next) */
                        cluevals[j] = (unsigned long)gcd_helper((long)cluevals[j], grid[i]);
                        break;
                    case C_LCM:
                        /* LCM: accumulate lcm(current, next) */
                        cluevals[j] = (unsigned long)lcm_helper((long)cluevals[j], grid[i]);
                        break;
                    case C_XOR:
                        /* XOR: accumulate xor(current, next) - self-inverse */
                        cluevals[j] ^= grid[i];
                        break;
                }
            }
        }

        /*
         * NOTE: Modular mode clue value wrapping is DISABLED.
         * See comments in new_game_desc() for details.
         */

        for (i = 0; i < a; i++) {
            j = dsf_canonify(dsf, i);
            if (j == i) {
                clues[j] |= cluevals[j];
            }
        }

        /*
         * See if the game can be solved at the specified difficulty.
         * Use cage merging to elevate difficulty when too easy.
         */
        if (diff > 0) {
            memset(soln, 0, (size_t)a * sizeof(digit));
            ret = keen_solver(w, dsf, clues, soln, diff - 1, params->mode_flags);
            if (ret <= diff - 1) {
                /* Too easy - try cage merging */
                int merge_attempts = 0;
                int max_merges = w * 2;

                while (merge_attempts < max_merges && ret <= diff - 1) {
                    if (!try_merge_cages(w, dsf, grid, clues, cluevals, maxblk, rs)) {
                        break;
                    }
                    merge_attempts++;
                    memset(soln, 0, (size_t)a * sizeof(digit));
                    ret = keen_solver(w, dsf, clues, soln, diff - 1, params->mode_flags);
                }

                if (ret <= diff - 1) continue;
            }
        }
        memset(soln, 0, (size_t)a * sizeof(digit));
        ret = keen_solver(w, dsf, clues, soln, diff, params->mode_flags);
        if (ret != diff) {
            if (ret < diff) {
                /* Too easy - merge cages */
                int merge_attempts = 0;
                int max_merges = w * 2;

                while (merge_attempts < max_merges && ret < diff) {
                    if (!try_merge_cages(w, dsf, grid, clues, cluevals, maxblk, rs)) {
                        break;
                    }
                    merge_attempts++;
                    memset(soln, 0, (size_t)a * sizeof(digit));
                    ret = keen_solver(w, dsf, clues, soln, diff, params->mode_flags);
                }
            }

            if (ret != diff) continue;
        }

        if (memcmp(soln, grid, (size_t)a * sizeof(digit)) != 0) {
            /* Solver found a different solution; reject this cage layout. */
            continue;
        }

        /* Success! */
        break;
    }

    /*
     * No fallback - user expects the exact difficulty they selected.
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
        return nullptr;
    }

    /* Encode description */
    desc = snewn(40 * a, char);
    p = desc;

    for (i = 0; i < a; i++) {
        j = dsf_canonify(dsf, i);
        p += sprintf(p, "%02d", j);
        if (i < a - 1) *p++ = ',';
    }

    *p++ = ';';

    for (i = 0; i < a; i++) {
        j = dsf_canonify(dsf, i);
        if (j == i) {
            switch ((unsigned long)clues[j] & CMASK) {
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
                    *p++ = 'o'; /* 'o' for m'o'dulo */
                    break;
                case C_GCD:
                    *p++ = 'g';
                    break;
                case C_LCM:
                    *p++ = 'l';
                    break;
                case C_XOR:
                    *p++ = 'x'; /* XOR: x ⊕ y notation */
                    break;
            }
            p += sprintf(p, "%05" PRIu64, (uint64_t)(clues[j] & ~CMASK));
            *p++ = ',';
        }
    }

    --p;
    *p++ = '\0';

    desc = sresize(desc, p - desc, char);

    /* Encode solution with mode-specific transformations */
    if (memcmp(soln, grid, (size_t)a * sizeof(digit)) != 0) {
        sfree(grid);
        sfree(order);
        sfree(revorder);
        sfree(singletons);
        sfree(dsf);
        sfree(clues);
        sfree(cluevals);
        sfree(soln);
        return nullptr;
    }
    *aux = snewn(a + 2, char);

    char* auxp = *aux;
    *auxp++ = 'S'; /* Solution marker */
    for (i = 0; i < a; i++) {
        int display_val = soln[i];

        if (HAS_MODE(params->mode_flags, MODE_ZERO_INCLUSIVE)) {
            display_val -= 1; /* 1..N -> 0..N-1 */
        } else if (HAS_MODE(params->mode_flags, MODE_NEGATIVE)) {
            int range = w;
            int half = range / 2;
            display_val -= (half + 1);
        }

        if (display_val >= 10)
            *auxp++ = (char)('A' + (display_val - 10));
        else if (display_val >= 0)
            *auxp++ = (char)('0' + display_val);
        else
            *auxp++ = '?';
    }
    *auxp = '\0';
    *aux = sresize(*aux, auxp - *aux + 1, char);

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

