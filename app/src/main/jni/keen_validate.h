/*
 * kenken_validate.h: Real-time validation for KenKen puzzles
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 *
 * Provides cell-level error detection for UI error highlighting.
 * Checks:
 *   - Row uniqueness (Latin square constraint)
 *   - Column uniqueness (Latin square constraint)
 *   - Cage constraint satisfaction (arithmetic operations)
 */

#ifndef KENKEN_VALIDATE_H
#define KENKEN_VALIDATE_H

#include "latin.h"

/*
 * Error flags for each cell. Multiple flags can be set simultaneously.
 * These are designed for efficient bitwise OR in the UI layer.
 */
#define VALID_OK 0x00       /* No errors detected */
#define VALID_ERR_ROW 0x01  /* Duplicate in row */
#define VALID_ERR_COL 0x02  /* Duplicate in column */
#define VALID_ERR_CAGE 0x04 /* Cage constraint violated */

/*
 * Validation context passed from JNI layer.
 * Encapsulates all puzzle state needed for validation.
 */
typedef struct {
    int w;          /* Grid width/height */
    digit* grid;    /* Current cell values (0 = empty) */
    int* dsf;       /* Disjoint set forest for cage membership */
    clue_t* clues;    /* Cage clues with operation encoded in upper bits */
    int mode_flags; /* Mode flags for special rules (e.g., MODE_KILLER) */
} validate_ctx;

/*
 * Validate entire grid and return error flags for each cell.
 *
 * Parameters:
 *   ctx    - Validation context with puzzle state
 *   errors - Output array of size w*w, filled with VALID_ERR_* flags
 *
 * Returns:
 *   Total number of cells with errors (0 = puzzle is valid so far)
 */
int kenken_validate_grid(const validate_ctx* ctx, int* errors);

/*
 * Check if a specific cell has any errors.
 * More efficient than full grid validation for single-cell updates.
 *
 * Parameters:
 *   ctx  - Validation context with puzzle state
 *   cell - Cell index (row * w + col)
 *
 * Returns:
 *   Bitmask of VALID_ERR_* flags for this cell
 */
int kenken_validate_cell(const validate_ctx* ctx, int cell);

/*
 * Check if puzzle is complete and valid.
 *
 * Parameters:
 *   ctx - Validation context with puzzle state
 *
 * Returns:
 *   1 if puzzle is completely filled and all constraints satisfied
 *   0 otherwise
 */
int kenken_is_complete(const validate_ctx* ctx);

#endif /* KENKEN_VALIDATE_H */
