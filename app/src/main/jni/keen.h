/*
 * kenken.h: Public API for KenKen puzzle generation
 *
 * Exposes the core puzzle generation functions to the JNI bridge layer.
 * Supports both random generation and ML-assisted generation from
 * pre-computed Latin squares.
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2016 Sergey
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 */

#ifndef KEENKENNING_KEEN_H
#define KEENKENNING_KEEN_H

#include "latin.h"
#include "puzzles.h"
#include "keen_modes.h"

struct game_params {
    int w, diff, multiplication_only;
    int mode_flags;  /* Bit flags from kenken_modes.h for extended modes */
};

char *new_game_desc(const game_params *params, random_state *rs, char **aux, int interactive);

char *new_game_desc_from_grid(const game_params *params, random_state *rs, digit *input_grid,
                              char **aux, int interactive);

#endif // KEENKENNING_KEEN_H
