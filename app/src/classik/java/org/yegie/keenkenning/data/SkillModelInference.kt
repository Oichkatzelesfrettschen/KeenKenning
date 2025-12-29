/*
 * SkillModelInference.kt: Heuristic skill prediction for adaptive difficulty (Classik)
 *
 * Classik flavor: Uses heuristic calculation only (no ML/ONNX).
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning.data

import android.content.Context
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Input features for skill prediction.
 */
data class SkillFeatures(
    val solveTimeRatio: Float,
    val errorRate: Float,
    val hintRate: Float,
    val undoRate: Float,
    val streak: Int,
    val puzzleSize: Int,
    val difficulty: Int,
    val sessionPuzzles: Int
) {
    fun toNormalizedArray(): FloatArray {
        return floatArrayOf(
            normalize(solveTimeRatio, 0f, 3f),
            normalize(errorRate, 0f, 0.5f),
            normalize(hintRate, 0f, 1f),
            normalize(undoRate, 0f, 0.5f),
            normalize(streak.toFloat(), 0f, 20f),
            normalize(puzzleSize.toFloat(), 3f, 16f),
            normalize(difficulty.toFloat(), 0f, 4f),
            normalize(sessionPuzzles.toFloat(), 0f, 50f)
        )
    }

    private fun normalize(value: Float, minVal: Float, maxVal: Float): Float {
        val normalized = (value - minVal) / (maxVal - minVal + 1e-8f)
        return normalized.coerceIn(0f, 1f)
    }
}

/**
 * Heuristic skill inference for Classik flavor.
 *
 * Thread-safe: can be called from any thread.
 */
class SkillModelInference(private val context: Context) {

    companion object {
        @Volatile
        @android.annotation.SuppressLint("StaticFieldLeak")
        private var instance: SkillModelInference? = null

        fun getInstance(context: Context): SkillModelInference {
            return instance ?: synchronized(this) {
                instance ?: SkillModelInference(context.applicationContext).also {
                    instance = it
                }
            }
        }
    }

    /**
     * Predict skill score from features using heuristics.
     */
    fun predictSkill(features: SkillFeatures): Float {
        return predictHeuristic(features)
    }

    /**
     * Heuristic skill calculation.
     * Mirrors the logic in UserStatsManager.computeSkillScore().
     */
    private fun predictHeuristic(features: SkillFeatures): Float {
        // Time factor (40%): faster = higher
        val timeFactor = (1f / max(features.solveTimeRatio, 0.1f)).coerceIn(0.2f, 2f)

        // Error penalty (15%)
        val errorPenalty = min(features.errorRate * 2f, 0.4f)

        // Hint penalty (15%)
        val hintPenalty = min(features.hintRate * 0.15f, 0.5f)

        // Undo penalty (10%)
        val undoPenalty = min(features.undoRate * 1.5f, 0.3f)

        // Difficulty bonus (20%)
        val difficultyBonus = features.difficulty * 0.2f

        // Streak bonus (soft cap at +0.15)
        val streakBonus = (1f - exp(-features.streak * 0.1f)) * 0.15f

        // Weighted combination
        val rawScore = (timeFactor * 0.4f) +
                       (difficultyBonus * 0.2f) +
                       (streakBonus) -
                       (hintPenalty * 0.15f) -
                       (errorPenalty * 0.15f) -
                       (undoPenalty * 0.1f)

        return rawScore.coerceIn(0f, 1f)
    }

    /**
     * Map skill score to recommended difficulty level.
     */
    fun recommendDifficulty(skillScore: Float, gridSize: Int): Int {
        val sizeAdjustment = when {
            gridSize >= 12 -> -0.1f
            gridSize <= 4 -> 0.1f
            else -> 0f
        }

        val adjusted = (skillScore + sizeAdjustment).coerceIn(0f, 1f)

        return when {
            adjusted < 0.25f -> 0  // Easy
            adjusted < 0.45f -> 1  // Normal
            adjusted < 0.65f -> 2  // Hard
            adjusted < 0.85f -> 3  // Insane
            else -> 4             // Ludicrous
        }
    }

    fun close() {
        instance = null
    }
}
