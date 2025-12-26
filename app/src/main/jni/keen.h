/*
 * keen.h: Android-specific header for KenKen puzzle generation
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: Copyright (C) 2016 Sergey
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKeen Contributors
 *
 * This file is part of KeenKeen for Android.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef KEEN_FOR_ANDROID_KEEN_H
#define KEEN_FOR_ANDROID_KEEN_H

#include "latin.h"
#include "puzzles.h"
#include "keen_modes.h"

struct game_params {
    int w, diff, multiplication_only;
    int mode_flags;  /* Bit flags from keen_modes.h for extended modes */
};

char *new_game_desc(const game_params *params, random_state *rs, char **aux, int interactive);

char *new_game_desc_from_grid(const game_params *params, random_state *rs, digit *input_grid,
                              char **aux, int interactive);

#endif // KEEN_FOR_ANDROID_KEEN_H
