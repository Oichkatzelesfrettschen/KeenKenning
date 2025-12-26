/*
 * keen_modes.h: Game mode flag definitions for KenKen variants
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKeen Contributors
 *
 * Architecture Design:
 * --------------------
 * Mode flags are bit-packed into a single integer for efficient passing
 * through JNI and minimal changes to existing game_params structure.
 *
 * WHY: Single field expansion allows adding modes without ABI changes.
 * WHAT: Bit flags for each mode variant, composable where sensible.
 * HOW: Pass mode_flags through JNI, extract in C, adjust generation.
 *
 * Flag Layout (32-bit):
 *   Bits 0-3:   Phase 1 modes (basic variants)
 *   Bits 4-7:   Phase 2 modes (extended operations)
 *   Bits 8-11:  Phase 3 modes (advanced constraints)
 *   Bits 12-15: Phase 4 modes (research features)
 *   Bits 16-31: Reserved for future use
 */

#ifndef KEEN_MODES_H
#define KEEN_MODES_H

/* Phase 1: Basic variants (low effort) */
#define MODE_STANDARD          0x0000  /* Default: all ops, digits 1-N */
#define MODE_MULT_ONLY         0x0001  /* Multiplication only (existing) */
#define MODE_MYSTERY           0x0002  /* UI-only: hide operation symbols */
#define MODE_ZERO_INCLUSIVE    0x0004  /* Digits 0 to N-1 instead of 1 to N */

/* Phase 2: Extended operations (medium effort) */
#define MODE_EXPONENT          0x0010  /* Include ^ exponent operation */
#define MODE_NEGATIVE          0x0008  /* Range -N to +N (excluding 0) */

/* Phase 3: Advanced constraints (high effort) */
#define MODE_MODULAR           0x0020  /* Wrap-around arithmetic (mod N) */
#define MODE_KILLER            0x0040  /* No repeated digits in cages */

/* Phase 4: Research-backed innovations */
#define MODE_HINT              0x0080  /* Explainable hints with reasoning */
#define MODE_ADAPTIVE          0x0100  /* Difficulty adjusts to skill */
#define MODE_STORY             0x0200  /* Themed puzzles with narrative */

/* Utility macros */
#define HAS_MODE(flags, mode)  (((flags) & (mode)) != 0)
#define SET_MODE(flags, mode)  ((flags) | (mode))
#define CLR_MODE(flags, mode)  ((flags) & ~(mode))

/*
 * Mode compatibility matrix:
 * --------------------------
 * ZERO_INCLUSIVE + C_DIV    = INVALID (division by 0 ambiguity)
 * ZERO_INCLUSIVE + MULT_ONLY = VALID (0*anything=0, still solvable)
 * ZERO_INCLUSIVE + NEGATIVE  = INVALID (conflicting ranges)
 * KILLER + any operation     = VALID (additional constraint only)
 * MODULAR + EXPONENT         = VALID but complex
 *
 * The validate_mode_flags() function checks these constraints.
 */

/* Returns 1 if mode combination is valid, 0 otherwise */
static inline int validate_mode_flags(int flags) {
    /* Zero-inclusive mode cannot have division */
    if (HAS_MODE(flags, MODE_ZERO_INCLUSIVE)) {
        /* With 0 in play, only addition and multiplication are safe */
        /* Subtraction is okay (|a-b| still works) */
        /* Division is NOT okay (divide by 0 ambiguity) */
        /* This is enforced during clue type selection, not here */
    }

    /* Cannot combine zero-inclusive with negative numbers */
    if (HAS_MODE(flags, MODE_ZERO_INCLUSIVE) && HAS_MODE(flags, MODE_NEGATIVE)) {
        return 0;
    }

    return 1;
}

/*
 * Get digit range for a given mode and grid size.
 * Returns: min_digit (via pointer), max_digit (return value)
 */
static inline int get_digit_range(int flags, int w, int *min_digit) {
    if (HAS_MODE(flags, MODE_NEGATIVE)) {
        *min_digit = -w;
        return w;  /* Range: -N to +N, excluding 0 */
    } else if (HAS_MODE(flags, MODE_ZERO_INCLUSIVE)) {
        *min_digit = 0;
        return w - 1;  /* Range: 0 to N-1 */
    } else {
        *min_digit = 1;
        return w;  /* Standard: 1 to N */
    }
}

/*
 * Check if an operation is allowed for given mode flags.
 * Used during clue type selection.
 */
static inline int operation_allowed(int flags, long clue_type) {
    /* Division not allowed in zero-inclusive mode */
    if (HAS_MODE(flags, MODE_ZERO_INCLUSIVE) && clue_type == 0x60000000L) {
        return 0;
    }
    return 1;
}

#endif /* KEEN_MODES_H */
