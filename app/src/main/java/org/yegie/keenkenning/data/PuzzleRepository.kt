/*
 * PuzzleRepository.kt: Repository for puzzle generation and data access
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning.data

import android.content.Context
import org.yegie.keenkenning.KeenModel
import org.yegie.keenkenning.NeuralKeenGenerator

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
            val builder = org.yegie.keenkenning.KeenModelBuilder()
            // Always compute ML probabilities for Smart Hints availability
            // The 'useML' parameter now only affects tracking (isMlGenerated badge)
            // Pass game mode flags to C layer for mode-specific generation
            builder.build(context, size, diff, multOnly, seed, true, gameMode.cFlags)
        }
    }
}
