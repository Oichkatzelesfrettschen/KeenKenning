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
 * substantially extended for Keen Kenning with additional game modes,
 * AI-assisted generation, and Android JNI integration.
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2004-2024 Simon Tatham
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
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

#include "keen.h"
#include "keen_internal.h"
#include "keen_solver.h"

#include <assert.h>
#include <ctype.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Difficulty levels: X-macro pattern ensures the enum values and
 * their string representations stay synchronized automatically.
 */
#define TITLE(upper, title, func, lower) #title,
#define ENCODE(upper, title, func, lower) #lower
#define CONFIG(upper, title, func, lower) ":" #title
static char const* const keen_diffnames[] = {DIFFLIST(TITLE)};
static char const keen_diffchars[] = DIFFLIST(ENCODE);
#define DIFFCONFIG DIFFLIST(CONFIG)

/*
 * Clue notation: Operation codes stored in high bits of clue value.
 * Ordering matters: ADD/MUL before SUB/DIV ensures preference during
 * generation when multiple operations yield the same result.
 *
 * 4-bit encoding (bits 28-31) with backward-compatible layout:
 * - EVEN slots (0,2,4,6,8,10,12,14): Original 8 operations
 * - ODD slots (1,3,5,7,...): Extended operations (bitwise, etc.)
 *
 * Original operations (even slots):
 *   0x0: ADD  - Sum of all digits
 *   0x2: MUL  - Product of all digits
 *   0x4: SUB  - Absolute difference (2-cell only)
 *   0x6: DIV  - Integer quotient (2-cell only)
 *   0x8: EXP  - Exponentiation base^exp (2-cell, requires MODE_EXPONENT)
 *   0xA: MOD  - Modulo operation (2-cell, requires MODE_NUMBER_THEORY)
 *   0xC: GCD  - Greatest common divisor (requires MODE_NUMBER_THEORY)
 *   0xE: LCM  - Least common multiple (requires MODE_NUMBER_THEORY)
 *
 * Extended operations (odd slots, requires MODE_BITWISE):
 *   0x1: XOR  - Bitwise exclusive-or (high ambiguity for difficulty)
 *   0x3: AND  - Bitwise and (reserved)
 *   0x5: OR   - Bitwise or (reserved)
 *   0x7-0xF: Reserved for future operations
 */
enum { COL_BACKGROUND, COL_GRID, COL_USER, COL_HIGHLIGHT, COL_ERROR, COL_PENCIL, NCOLOURS };

struct clues {
    int refcount;
    int w;
    int* dsf;
    clue_t* clues;
};

struct game_state {
    game_params par;
    struct clues* clues;
    digit* grid;
    int* pencil; /* bitmaps using bits 1<<1..1<<n */
    int completed, cheated;
};

[[maybe_unused]] static game_params* default_params(void) {
    game_params* ret = snew(game_params);

    ret->w = 6;
    ret->diff = DIFF_NORMAL;
    ret->multiplication_only = false;
    ret->mode_flags = MODE_STANDARD;

    return ret;
}

const static struct game_params keen_presets[] = {
    {3, DIFF_EASY, false, MODE_STANDARD},         {4, DIFF_EASY, false, MODE_STANDARD},
    {5, DIFF_NORMAL, false, MODE_STANDARD},       {6, DIFF_NORMAL, false, MODE_STANDARD},
    {7, DIFF_HARD, false, MODE_STANDARD},         {8, DIFF_EXTREME, false, MODE_STANDARD},
    {9, DIFF_UNREASONABLE, false, MODE_STANDARD},
};

[[maybe_unused]] static int game_fetch_preset(int i, char** name, game_params** params) {
    game_params* ret;
    char buf[80];

    if (i < 0 || (size_t)i >= lenof(keen_presets)) return false;

    ret = snew(game_params);
    *ret = keen_presets[i]; /* structure copy */

    sprintf(buf, "%dx%d %s%s", ret->w, ret->w, keen_diffnames[ret->diff],
            ret->multiplication_only ? ", multiplication only" : "");

    *name = dupstr(buf);
    *params = ret;
    return true;
}

[[maybe_unused]] static void free_params(game_params* params) {
    sfree(params);
}

[[maybe_unused]] static game_params* dup_params(const game_params* params) {
    game_params* ret = snew(game_params);
    *ret = *params; /* structure copy */
    return ret;
}

[[maybe_unused]] static void decode_params(game_params* params, char const* string) {
    char const* p = string;

    params->w = atoi(p);
    while (*p && isdigit((unsigned char)*p)) p++;

    if (*p == 'd') {
        int i;
        p++;
        params->diff = DIFFCOUNT + 1; /* ...which is invalid */
        if (*p) {
            for (i = 0; i < DIFFCOUNT; i++) {
                if (*p == keen_diffchars[i]) params->diff = i;
            }
            p++;
        }
    }

    if (*p == 'm') {
        p++;
        params->multiplication_only = true;
    }
}

[[maybe_unused]] static char* encode_params(const game_params* params, int full) {
    char ret[80];

    sprintf(ret, "%d", params->w);
    if (full)
        sprintf(ret + strlen(ret), "d%c%s", keen_diffchars[params->diff],
                params->multiplication_only ? "m" : "");

    return dupstr(ret);
}

[[maybe_unused]] static config_item* game_configure(const game_params* params) {
    config_item* ret;
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
    ret[2].sval = nullptr;
    ret[2].ival = params->multiplication_only;

    ret[3].name = nullptr;
    ret[3].type = C_END;
    ret[3].sval = nullptr;
    ret[3].ival = 0;

    return ret;
}

[[maybe_unused]] static game_params* custom_params(const config_item* cfg) {
    game_params* ret = snew(game_params);

    ret->w = atoi(cfg[0].sval);
    ret->diff = cfg[1].ival;
    ret->multiplication_only = cfg[2].ival;
    ret->mode_flags = MODE_STANDARD;

    return ret;
}

[[maybe_unused]] static char* validate_params(const game_params* params, [[maybe_unused]] int full) {
    if (params->w < 3 || params->w > 16) return "Grid size must be between 3 and 16";
    if (params->diff >= DIFFCOUNT) return "Unknown difficulty rating";
    return nullptr;
}

/* ======================================================================
 * LEGACY DESKTOP GUI CODE REMOVED
 * ======================================================================
 *
 * The following functionality from Simon Tatham's original desktop
 * implementation has been removed as Keen Kenning uses Jetpack Compose
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
