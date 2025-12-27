/*
 * kenken_validate.c: Real-time validation for KenKen puzzles
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 *
 * Implements efficient validation for real-time error highlighting.
 * Row/column checks use bitmask counting for O(n) per row/column.
 * Cage checks evaluate arithmetic operations on filled cells.
 */

#include "kenken_validate.h"
#include "kenken_modes.h"
#include "puzzles.h"  /* For smalloc, sfree */
#include "latin.h"    /* For digit type */
#include <string.h>

/* Maximum cage size - must match kenken.c */
#define MAXBLK 6

/* Operation codes - must match kenken.c */
#define C_ADD 0x00000000L
#define C_MUL 0x20000000L
#define C_SUB 0x40000000L
#define C_DIV 0x60000000L
#define C_EXP 0x80000000L
#define C_MOD 0xA0000000L
#define C_GCD 0xC0000000L
#define C_LCM 0xE0000000L
#define CMASK 0xE0000000L

/* Forward declaration for dsf_canonify from dsf.c */
extern int dsf_canonify(int *dsf, int index);

/*
 * GCD helper using Euclidean algorithm.
 */
static long gcd(long a, long b) {
    while (b != 0) {
        long t = b;
        b = a % b;
        a = t;
    }
    return a < 0 ? -a : a;
}

/*
 * LCM helper - divide first to avoid overflow.
 */
static long lcm(long a, long b) {
    if (a == 0 || b == 0) return 0;
    long g = gcd(a, b);
    return (a / g) * b;
}

/*
 * Check a single row for duplicates.
 * Returns bitmask of cell indices within the row that have duplicates.
 */
static int check_row_duplicates(const validate_ctx *ctx, int row, int *errors) {
    int w = ctx->w;
    int seen[17] = {0};  /* seen[digit] = first occurrence index + 1, or 0 if unseen */
    int count = 0;

    for (int col = 0; col < w; col++) {
        int cell = row * w + col;
        digit d = ctx->grid[cell];
        if (d == 0) continue;  /* Empty cell */

        if (seen[d]) {
            /* Duplicate found - mark both cells */
            int first_cell = row * w + (seen[d] - 1);
            errors[first_cell] |= VALID_ERR_ROW;
            errors[cell] |= VALID_ERR_ROW;
            count++;
        } else {
            seen[d] = col + 1;  /* Store column + 1 (so 0 means unseen) */
        }
    }
    return count;
}

/*
 * Check a single column for duplicates.
 */
static int check_col_duplicates(const validate_ctx *ctx, int col, int *errors) {
    int w = ctx->w;
    int seen[17] = {0};
    int count = 0;

    for (int row = 0; row < w; row++) {
        int cell = row * w + col;
        digit d = ctx->grid[cell];
        if (d == 0) continue;

        if (seen[d]) {
            int first_cell = (seen[d] - 1) * w + col;
            errors[first_cell] |= VALID_ERR_COL;
            errors[cell] |= VALID_ERR_COL;
            count++;
        } else {
            seen[d] = row + 1;
        }
    }
    return count;
}

/*
 * Collect all cells belonging to a cage and their values.
 * Returns number of filled cells (-1 if cage is incomplete).
 */
static int collect_cage(const validate_ctx *ctx, int root,
                        int *cells, digit *values, int *total_cells) {
    int w = ctx->w;
    int n = w * w;
    int filled = 0;
    *total_cells = 0;

    for (int i = 0; i < n; i++) {
        if (dsf_canonify(ctx->dsf, i) == root) {
            cells[*total_cells] = i;
            values[*total_cells] = ctx->grid[i];
            if (ctx->grid[i] != 0) filled++;
            (*total_cells)++;
        }
    }
    return filled;
}

/*
 * Check if cage values satisfy the arithmetic constraint.
 * Only called when all cells in the cage are filled.
 * Returns 1 if satisfied, 0 if violated.
 */
static int check_cage_constraint(const validate_ctx *ctx, int root,
                                  digit *values, int ncells) {
    long clue = ctx->clues[root];
    long target = clue & ~CMASK;
    long op = clue & CMASK;
    int i;

    /* Killer mode: check for duplicate digits in cage */
    if (ctx->mode_flags & MODE_KILLER) {
        int seen = 0;
        for (i = 0; i < ncells; i++) {
            int bit = 1 << values[i];
            if (seen & bit) return 0;  /* Duplicate in cage */
            seen |= bit;
        }
    }

    switch (op) {
        case C_ADD: {
            long sum = 0;
            for (i = 0; i < ncells; i++) sum += values[i];
            return sum == target;
        }

        case C_MUL: {
            long prod = 1;
            for (i = 0; i < ncells; i++) prod *= values[i];
            return prod == target;
        }

        case C_SUB: {
            /* 2-cell only: |a - b| = target */
            if (ncells != 2) return 0;
            long diff = values[0] - values[1];
            if (diff < 0) diff = -diff;
            return diff == target;
        }

        case C_DIV: {
            /* 2-cell only: max/min = target (exact division) */
            if (ncells != 2) return 0;
            long a = values[0], b = values[1];
            if (a < b) { long t = a; a = b; b = t; }
            if (b == 0) return 0;
            return (a % b == 0) && (a / b == target);
        }

        case C_EXP: {
            /* 2-cell only: a^b = target or b^a = target */
            if (ncells != 2) return 0;
            long a = values[0], b = values[1];
            long pow_ab = 1, pow_ba = 1;
            for (i = 0; i < b && pow_ab <= target; i++) pow_ab *= a;
            for (i = 0; i < a && pow_ba <= target; i++) pow_ba *= b;
            return (pow_ab == target) || (pow_ba == target);
        }

        case C_MOD: {
            /* 2-cell only: a % b = target or b % a = target */
            if (ncells != 2) return 0;
            long a = values[0], b = values[1];
            if (a == 0 && b == 0) return target == 0;
            return (b != 0 && a % b == target) || (a != 0 && b % a == target);
        }

        case C_GCD: {
            long g = values[0];
            for (i = 1; i < ncells; i++) g = gcd(g, values[i]);
            return g == target;
        }

        case C_LCM: {
            long l = values[0];
            for (i = 1; i < ncells; i++) {
                l = lcm(l, values[i]);
                if (l > 10000000) return 0;  /* Overflow protection */
            }
            return l == target;
        }

        default:
            return 1;  /* Unknown op - assume valid */
    }
}

/*
 * Validate entire grid.
 */
int kenken_validate_grid(const validate_ctx *ctx, int *errors) {
    int w = ctx->w;
    int n = w * w;
    int row, col, i;

    /* Clear error array */
    memset(errors, 0, n * sizeof(int));

    /* Check rows */
    for (row = 0; row < w; row++) {
        check_row_duplicates(ctx, row, errors);
    }

    /* Check columns */
    for (col = 0; col < w; col++) {
        check_col_duplicates(ctx, col, errors);
    }

    /* Check cages - only fully filled ones */
    int *checked = (int *)smalloc(n * sizeof(int));
    memset(checked, 0, n * sizeof(int));

    digit values[MAXBLK + 1];
    int cells[MAXBLK + 1];

    for (i = 0; i < n; i++) {
        int root = dsf_canonify(ctx->dsf, i);
        if (checked[root]) continue;
        checked[root] = 1;

        int total_cells;
        int filled = collect_cage(ctx, root, cells, values, &total_cells);

        /* Only validate if cage is completely filled */
        if (filled == total_cells && total_cells > 0) {
            if (!check_cage_constraint(ctx, root, values, total_cells)) {
                /* Mark all cells in this cage with cage error */
                for (int j = 0; j < total_cells; j++) {
                    errors[cells[j]] |= VALID_ERR_CAGE;
                }
            }
        }
    }

    sfree(checked);

    /* Count actual cells with errors */
    int error_count = 0;
    for (i = 0; i < n; i++) {
        if (errors[i] != VALID_OK) error_count++;
    }

    return error_count;
}

/*
 * Validate a single cell (for incremental updates).
 */
int kenken_validate_cell(const validate_ctx *ctx, int cell) {
    int w = ctx->w;
    int row = cell / w;
    int col = cell % w;
    int result = VALID_OK;
    digit d = ctx->grid[cell];

    if (d == 0) return VALID_OK;  /* Empty cell has no errors */

    /* Check row */
    for (int c = 0; c < w; c++) {
        if (c != col && ctx->grid[row * w + c] == d) {
            result |= VALID_ERR_ROW;
            break;
        }
    }

    /* Check column */
    for (int r = 0; r < w; r++) {
        if (r != row && ctx->grid[r * w + col] == d) {
            result |= VALID_ERR_COL;
            break;
        }
    }

    /* Check cage constraint */
    int root = dsf_canonify(ctx->dsf, cell);
    int total_cells;
    digit values[MAXBLK + 1];
    int cells[MAXBLK + 1];

    int filled = collect_cage(ctx, root, cells, values, &total_cells);

    if (filled == total_cells && total_cells > 0) {
        if (!check_cage_constraint(ctx, root, values, total_cells)) {
            result |= VALID_ERR_CAGE;
        }
    }

    return result;
}

/*
 * Check if puzzle is complete and valid.
 */
int kenken_is_complete(const validate_ctx *ctx) {
    int w = ctx->w;
    int n = w * w;

    /* Check all cells are filled */
    for (int i = 0; i < n; i++) {
        if (ctx->grid[i] == 0) return 0;
    }

    /* Validate entire grid */
    int *errors = (int *)smalloc(n * sizeof(int));
    int error_count = kenken_validate_grid(ctx, errors);
    sfree(errors);

    return error_count == 0;
}

