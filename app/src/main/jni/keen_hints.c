/*
 * kenken_hints.c: Step-by-step hint system for KenKen puzzles
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 *
 * Implements hint generation using logical deduction techniques.
 * Hints are ordered by difficulty to provide natural learning progression.
 */

#include "keen_hints.h"

#include <string.h>

#include "keen_modes.h"
#include "puzzles.h"

/* Forward declaration */
extern int dsf_canonify(int* dsf, int index);

/*
 * Calculate candidates for a cell based on row/column constraints.
 * Returns a bitmask where bit i is set if value i is possible.
 */
static int get_candidates(const hint_ctx* ctx, int cell) {
    int w = ctx->w;
    int row = cell / w;
    int col = cell % w;
    int candidates = (1 << (w + 1)) - 2; /* Bits 1..w set */

    /* Eliminate values already in row */
    for (int c = 0; c < w; c++) {
        int other = row * w + c;
        if (ctx->grid[other] != 0) {
            candidates &= ~(1 << ctx->grid[other]);
        }
    }

    /* Eliminate values already in column */
    for (int r = 0; r < w; r++) {
        int other = r * w + col;
        if (ctx->grid[other] != 0) {
            candidates &= ~(1 << ctx->grid[other]);
        }
    }

    return candidates;
}

/*
 * Count bits set in a bitmask.
 */
static int popcount(int x) {
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

/*
 * Get the single set bit position (1-indexed).
 * Returns 0 if not exactly one bit is set.
 */
static int single_bit_pos(int x) {
    if (x == 0 || (x & (x - 1)) != 0) return 0; /* Not exactly one bit */
    int pos = 1;
    while ((x & (1 << pos)) == 0) pos++;
    return pos;
}

/*
 * Check for naked single: only one candidate for a cell.
 */
static int find_naked_single(const hint_ctx* ctx, hint_result* result) {
    int w = ctx->w;
    int n = w * w;

    for (int cell = 0; cell < n; cell++) {
        if (ctx->grid[cell] != 0) continue; /* Already filled */

        int candidates = get_candidates(ctx, cell);
        if (popcount(candidates) == 1) {
            int value = single_bit_pos(candidates);
            if (value > 0) {
                result->hint_type = HINT_NAKED_SINGLE;
                result->cell = cell;
                result->row = cell / w;
                result->col = cell % w;
                result->value = value;
                result->cage_root = dsf_canonify(ctx->dsf, cell);
                result->related_pos = -1;
                return 1;
            }
        }
    }
    return 0;
}

/*
 * Check for hidden single: value can only go in one cell in a row.
 */
static int find_hidden_single_row(const hint_ctx* ctx, hint_result* result) {
    int w = ctx->w;

    for (int row = 0; row < w; row++) {
        /* For each value 1..w */
        for (int val = 1; val <= w; val++) {
            /* Check if value already in row */
            int found = 0;
            for (int col = 0; col < w; col++) {
                if (ctx->grid[row * w + col] == val) {
                    found = 1;
                    break;
                }
            }
            if (found) continue;

            /* Count cells where this value can go */
            int possible_cell = -1;
            int count = 0;
            for (int col = 0; col < w; col++) {
                int cell = row * w + col;
                if (ctx->grid[cell] != 0) continue;

                int candidates = get_candidates(ctx, cell);
                if (candidates & (1 << val)) {
                    possible_cell = cell;
                    count++;
                }
            }

            if (count == 1 && possible_cell >= 0) {
                result->hint_type = HINT_HIDDEN_SINGLE;
                result->cell = possible_cell;
                result->row = row;
                result->col = possible_cell % w;
                result->value = val;
                result->cage_root = dsf_canonify(ctx->dsf, possible_cell);
                result->related_pos = row; /* The row that forces this */
                return 1;
            }
        }
    }
    return 0;
}

/*
 * Check for hidden single in column.
 */
static int find_hidden_single_col(const hint_ctx* ctx, hint_result* result) {
    int w = ctx->w;

    for (int col = 0; col < w; col++) {
        for (int val = 1; val <= w; val++) {
            /* Check if value already in column */
            int found = 0;
            for (int row = 0; row < w; row++) {
                if (ctx->grid[row * w + col] == val) {
                    found = 1;
                    break;
                }
            }
            if (found) continue;

            /* Count cells where this value can go */
            int possible_cell = -1;
            int count = 0;
            for (int row = 0; row < w; row++) {
                int cell = row * w + col;
                if (ctx->grid[cell] != 0) continue;

                int candidates = get_candidates(ctx, cell);
                if (candidates & (1 << val)) {
                    possible_cell = cell;
                    count++;
                }
            }

            if (count == 1 && possible_cell >= 0) {
                result->hint_type = HINT_HIDDEN_SINGLE;
                result->cell = possible_cell;
                result->row = possible_cell / w;
                result->col = col;
                result->value = val;
                result->cage_root = dsf_canonify(ctx->dsf, possible_cell);
                result->related_pos = col + 100; /* +100 to indicate column */
                return 1;
            }
        }
    }
    return 0;
}

/*
 * Check for single-cell cage (the clue IS the answer).
 */
static int find_cage_single(const hint_ctx* ctx, hint_result* result) {
    int w = ctx->w;
    int n = w * w;

    for (int cell = 0; cell < n; cell++) {
        if (ctx->grid[cell] != 0) continue;

        int root = dsf_canonify(ctx->dsf, cell);

        /* Count cells in this cage */
        int cage_size = 0;
        for (int i = 0; i < n; i++) {
            if (dsf_canonify(ctx->dsf, i) == root) cage_size++;
        }

        if (cage_size == 1) {
            /* Single-cell cage - clue value IS the answer */
            unsigned long clue = (unsigned long)(ctx->clues[root]);
            int value = (int)(clue & 0x1FFFFFFFL); /* Strip operation bits */

            if (value >= 1 && value <= w) {
                result->hint_type = HINT_CAGE_SINGLE;
                result->cell = cell;
                result->row = cell / w;
                result->col = cell % w;
                result->value = value;
                result->cage_root = root;
                result->related_pos = -1;
                return 1;
            }
        }
    }
    return 0;
}

/*
 * Find next hint using progressive difficulty.
 */
int kenken_get_hint(const hint_ctx* ctx, hint_result* result) {
    memset(result, 0, sizeof(hint_result));

    /* Priority 1: Single-cell cages (trivial) */
    if (find_cage_single(ctx, result)) return 1;

    /* Priority 2: Naked singles (only one candidate) */
    if (find_naked_single(ctx, result)) return 1;

    /* Priority 3: Hidden singles in rows */
    if (find_hidden_single_row(ctx, result)) return 1;

    /* Priority 4: Hidden singles in columns */
    if (find_hidden_single_col(ctx, result)) return 1;

    /* No simple hint available - puzzle may need guessing */
    result->hint_type = HINT_NONE;
    return 0;
}

/*
 * Explain a specific cell.
 */
int kenken_explain_cell(const hint_ctx* ctx, int cell, hint_result* result) {
    int w = ctx->w;

    if (cell < 0 || cell >= w * w) return 0;

    if (ctx->grid[cell] != 0) return 0; /* Already filled */

    memset(result, 0, sizeof(hint_result));

    /* Check if this is a single-cell cage */
    int root = dsf_canonify(ctx->dsf, cell);
    int cage_size = 0;
    for (int i = 0; i < w * w; i++) {
        if (dsf_canonify(ctx->dsf, i) == root) cage_size++;
    }

    if (cage_size == 1) {
        unsigned long clue = (unsigned long)(ctx->clues[root]);
        int value = (int)(clue & 0x1FFFFFFFL);
        if (value >= 1 && value <= w) {
            result->hint_type = HINT_CAGE_SINGLE;
            result->cell = cell;
            result->row = cell / w;
            result->col = cell % w;
            result->value = value;
            result->cage_root = root;
            return 1;
        }
    }

    /* Check for naked single */
    int candidates = get_candidates(ctx, cell);
    if (popcount(candidates) == 1) {
        result->hint_type = HINT_NAKED_SINGLE;
        result->cell = cell;
        result->row = cell / w;
        result->col = cell % w;
        result->value = single_bit_pos(candidates);
        result->cage_root = root;
        return 1;
    }

    /* Check if solution is known and cell can be deduced */
    if (ctx->solution && ctx->solution[cell] != 0) {
        int val = ctx->solution[cell];

        /* Verify this value is still a candidate */
        if (candidates & (1 << val)) {
            result->hint_type = HINT_CAGE_FORCE; /* Generic "trust me" */
            result->cell = cell;
            result->row = cell / w;
            result->col = cell % w;
            result->value = val;
            result->cage_root = root;
            return 1;
        }
    }

    return 0;
}
