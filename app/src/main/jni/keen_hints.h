/*
 * kenken_hints.h: Step-by-step hint system for KenKen puzzles
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 *
 * Provides progressive hints without revealing the complete solution:
 *   Level 1: "Look at this cage/row/column"
 *   Level 2: "This cell can be determined"
 *   Level 3: "The value is X"
 */

#ifndef KENKEN_HINTS_H
#define KENKEN_HINTS_H

#include "latin.h"

/*
 * Hint types - what reasoning technique leads to the solution
 */
#define HINT_NONE 0          /* No hint available (puzzle may be invalid) */
#define HINT_NAKED_SINGLE 1  /* Only one value fits in this cell */
#define HINT_HIDDEN_SINGLE 2 /* Value can only go in this cell in row/col */
#define HINT_CAGE_FORCE 3    /* Cage arithmetic forces this value */
#define HINT_CAGE_SINGLE 4   /* Single-cell cage with given value */

/*
 * Hint result structure.
 * Contains progressive information about how to solve a cell.
 */
typedef struct {
    int hint_type;   /* HINT_* constant indicating reasoning technique */
    int cell;        /* Cell index (row * w + col) */
    int row;         /* Row (0-indexed) */
    int col;         /* Column (0-indexed) */
    int value;       /* The value for this cell (1-N) */
    int cage_root;   /* Root cell of the relevant cage (for HINT_CAGE_*) */
    int related_pos; /* Related position (row/col index for HIDDEN_SINGLE) */
} hint_result;

/*
 * Hint context - puzzle state for hint computation.
 */
typedef struct {
    int w;           /* Grid width/height */
    digit* grid;     /* Current cell values (0 = empty) */
    int* dsf;        /* Disjoint set forest for cage membership */
    clue_t* clues;     /* Cage clues with operation in upper bits */
    int mode_flags;  /* Mode flags (e.g., MODE_KILLER) */
    digit* solution; /* Known solution (for verification) */
} hint_ctx;

/*
 * Find the next hint for the puzzle.
 *
 * Searches for cells that can be determined through logical deduction.
 * Returns the easiest-to-explain deduction available.
 *
 * Parameters:
 *   ctx    - Hint context with puzzle state and solution
 *   result - Output hint result (filled on success)
 *
 * Returns:
 *   1 if a hint was found, 0 if no logical deduction is available
 */
int kenken_get_hint(const hint_ctx* ctx, hint_result* result);

/*
 * Get hint for a specific cell.
 *
 * Explains why a particular empty cell has a specific value.
 * Useful when user clicks a cell asking "why this?"
 *
 * Parameters:
 *   ctx    - Hint context with puzzle state and solution
 *   cell   - Cell to explain
 *   result - Output hint result
 *
 * Returns:
 *   1 if explanation found, 0 if cell is already filled or no explanation
 */
int kenken_explain_cell(const hint_ctx* ctx, int cell, hint_result* result);

#endif /* KENKEN_HINTS_H */
