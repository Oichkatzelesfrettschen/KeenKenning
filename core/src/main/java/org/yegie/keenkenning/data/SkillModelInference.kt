/*
 * SkillModelInference.kt: Skill inference interface for adaptive difficulty
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning.data

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

interface SkillModelInference {
    fun predictSkill(features: SkillFeatures): Float

    fun recommendDifficulty(skillScore: Float, gridSize: Int): Int

    fun close()
}
