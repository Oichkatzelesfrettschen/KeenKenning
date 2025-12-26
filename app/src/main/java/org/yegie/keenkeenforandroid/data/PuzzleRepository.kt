/*
 * PuzzleRepository.kt: Repository for puzzle generation and data access
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKeen Contributors
 */

package org.yegie.keenkeenforandroid.data

import android.content.Context
import org.yegie.keenkeenforandroid.KeenModel
import org.yegie.keenkeenforandroid.NeuralKeenGenerator

interface PuzzleRepository {
    suspend fun generatePuzzle(
        context: Context, // Needed for Neural Generator asset loading
        size: Int,
        diff: Int,
        multOnly: Int,
        seed: Long,
        useAI: Boolean,
        gameMode: GameMode = GameMode.STANDARD
    ): KeenModel
}

class PuzzleRepositoryImpl : PuzzleRepository {

    override suspend fun generatePuzzle(
        context: Context,
        size: Int,
        diff: Int,
        multOnly: Int,
        seed: Long,
        useAI: Boolean,
        gameMode: GameMode
    ): KeenModel {
        // Run on IO dispatcher
        return kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
            val builder = org.yegie.keenkeenforandroid.KeenModelBuilder()
            // Always compute AI probabilities for Smart Hints availability
            // The 'useAI' parameter now only affects tracking (isAiGenerated badge)
            // Pass game mode flags to C layer for mode-specific generation
            builder.build(context, size, diff, multOnly, seed, true, gameMode.cFlags)
        }
    }
}
