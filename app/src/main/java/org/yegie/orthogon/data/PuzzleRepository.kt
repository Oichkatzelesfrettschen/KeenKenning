/*
 * PuzzleRepository.kt: Repository for puzzle generation and data access
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 */

package org.yegie.orthogon.data

import android.content.Context
import org.yegie.orthogon.KenKenModel
import org.yegie.orthogon.NeuralKenKenGenerator

interface PuzzleRepository {
    suspend fun generatePuzzle(
        context: Context, // Needed for Neural Generator asset loading
        size: Int,
        diff: Int,
        multOnly: Int,
        seed: Long,
        useAI: Boolean,
        gameMode: GameMode = GameMode.STANDARD
    ): KenKenModel
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
    ): KenKenModel {
        // Run on IO dispatcher
        return kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
            val builder = org.yegie.orthogon.KenKenModelBuilder()
            // Always compute ML probabilities for Smart Hints availability
            // The 'useML' parameter now only affects tracking (isMlGenerated badge)
            // Pass game mode flags to C layer for mode-specific generation
            builder.build(context, size, diff, multOnly, seed, true, gameMode.cFlags)
        }
    }
}
