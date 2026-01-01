/*
 * PuzzleRepository.kt: Repository for puzzle generation and data access
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning.data

import android.content.Context
import org.yegie.keenkenning.KeenModel

/**
 * Result of puzzle generation - either success with model or failure with error message.
 */
sealed class PuzzleResult {
    data class Success(val model: KeenModel) : PuzzleResult()
    data class Failure(val errorMessage: String) : PuzzleResult()
}

interface PuzzleRepository {
    suspend fun generatePuzzle(
        context: Context, // Needed for Neural Generator asset loading
        size: Int,
        diff: Int,
        multOnly: Int,
        seed: Long,
        useAI: Boolean,
        gameMode: GameMode = GameMode.STANDARD
    ): PuzzleResult
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
    ): PuzzleResult {
        // Run on IO dispatcher
        return kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
            val builder = org.yegie.keenkenning.KeenModelBuilder()
            // Always compute ML probabilities for Smart Hints availability
            // The 'useML' parameter now only affects tracking (isMlGenerated badge)
            // Pass game mode flags to C layer for mode-specific generation
            val model = builder.build(context, size, diff, multOnly, seed, true, gameMode.cFlags)

            if (model != null) {
                PuzzleResult.Success(model)
            } else {
                // Get error message from builder for user feedback
                val errorMsg = builder.lastJniError ?: "Puzzle generation failed"
                PuzzleResult.Failure(errorMsg)
            }
        }
    }
}
