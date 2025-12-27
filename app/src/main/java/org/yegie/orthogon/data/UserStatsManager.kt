/*
 * UserStatsManager.kt: Player performance tracking for Adaptive Mode (Phase 4b)
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 */

package org.yegie.orthogon.data

import android.content.Context
import android.content.SharedPreferences
import kotlin.math.max
import kotlin.math.min

/**
 * Player performance statistics for adaptive difficulty.
 * Tracks solve times, success rates, and hints used per grid size.
 */
data class PlayerStats(
    val totalPuzzlesSolved: Int = 0,
    val totalPuzzlesAbandoned: Int = 0,
    val totalHintsUsed: Int = 0,
    val averageSolveTimeSeconds: Long = 0,
    // Per-size stats: key = gridSize (3-9), value = (solveCount, avgTimeSeconds)
    val sizeStats: Map<Int, SizeStats> = emptyMap(),
    // Computed skill score: 0.0 (beginner) to 1.0 (expert)
    val skillScore: Float = 0.5f
)

data class SizeStats(
    val solveCount: Int = 0,
    val averageTimeSeconds: Long = 0,
    val bestTimeSeconds: Long = Long.MAX_VALUE
)

/**
 * Manages player statistics persistence for adaptive difficulty.
 */
class UserStatsManager(context: Context) {

    companion object {
        private const val PREFS_NAME = "orthogon_user_stats"
        private const val KEY_TOTAL_SOLVED = "total_solved"
        private const val KEY_TOTAL_ABANDONED = "total_abandoned"
        private const val KEY_TOTAL_HINTS = "total_hints"
        private const val KEY_AVG_TIME = "avg_time"
        private const val KEY_SKILL_SCORE = "skill_score"
        private const val KEY_SIZE_STATS_PREFIX = "size_stats_"

        // Target solve times per grid size (seconds) for "average" player
        // Used to calibrate skill score
        private val TARGET_TIMES = mapOf(
            3 to 30L,
            4 to 60L,
            5 to 120L,
            6 to 240L,
            7 to 360L,
            8 to 480L,
            9 to 600L
        )
    }

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun getStats(): PlayerStats {
        val sizeStats = mutableMapOf<Int, SizeStats>()
        for (size in 3..9) {
            val key = "$KEY_SIZE_STATS_PREFIX$size"
            val count = prefs.getInt("${key}_count", 0)
            if (count > 0) {
                sizeStats[size] = SizeStats(
                    solveCount = count,
                    averageTimeSeconds = prefs.getLong("${key}_avg", 0),
                    bestTimeSeconds = prefs.getLong("${key}_best", Long.MAX_VALUE)
                )
            }
        }

        return PlayerStats(
            totalPuzzlesSolved = prefs.getInt(KEY_TOTAL_SOLVED, 0),
            totalPuzzlesAbandoned = prefs.getInt(KEY_TOTAL_ABANDONED, 0),
            totalHintsUsed = prefs.getInt(KEY_TOTAL_HINTS, 0),
            averageSolveTimeSeconds = prefs.getLong(KEY_AVG_TIME, 0),
            sizeStats = sizeStats,
            skillScore = prefs.getFloat(KEY_SKILL_SCORE, 0.5f)
        )
    }

    /**
     * Record a completed puzzle for stats tracking.
     * @param gridSize The puzzle grid size (3-9)
     * @param solveTimeSeconds Time taken to solve
     * @param hintsUsed Number of hints used
     * @param difficulty Original difficulty level (0-4)
     */
    fun recordPuzzleSolved(gridSize: Int, solveTimeSeconds: Long, hintsUsed: Int, difficulty: Int) {
        val editor = prefs.edit()

        // Update global stats
        val newTotal = prefs.getInt(KEY_TOTAL_SOLVED, 0) + 1
        val oldAvg = prefs.getLong(KEY_AVG_TIME, 0)
        val newAvg = if (newTotal == 1) solveTimeSeconds
                     else (oldAvg * (newTotal - 1) + solveTimeSeconds) / newTotal
        val newHints = prefs.getInt(KEY_TOTAL_HINTS, 0) + hintsUsed

        editor.putInt(KEY_TOTAL_SOLVED, newTotal)
        editor.putLong(KEY_AVG_TIME, newAvg)
        editor.putInt(KEY_TOTAL_HINTS, newHints)

        // Update per-size stats
        val key = "$KEY_SIZE_STATS_PREFIX$gridSize"
        val sizeCount = prefs.getInt("${key}_count", 0) + 1
        val sizeOldAvg = prefs.getLong("${key}_avg", 0)
        val sizeNewAvg = if (sizeCount == 1) solveTimeSeconds
                         else (sizeOldAvg * (sizeCount - 1) + solveTimeSeconds) / sizeCount
        val sizeBest = min(prefs.getLong("${key}_best", Long.MAX_VALUE), solveTimeSeconds)

        editor.putInt("${key}_count", sizeCount)
        editor.putLong("${key}_avg", sizeNewAvg)
        editor.putLong("${key}_best", sizeBest)

        // Recompute skill score
        val skillScore = computeSkillScore(gridSize, solveTimeSeconds, hintsUsed, difficulty)
        val oldSkill = prefs.getFloat(KEY_SKILL_SCORE, 0.5f)
        // Exponential moving average: 20% new, 80% old for stability
        val newSkill = oldSkill * 0.8f + skillScore * 0.2f
        editor.putFloat(KEY_SKILL_SCORE, newSkill.coerceIn(0.1f, 0.95f))

        editor.apply()
    }

    fun recordPuzzleAbandoned() {
        prefs.edit()
            .putInt(KEY_TOTAL_ABANDONED, prefs.getInt(KEY_TOTAL_ABANDONED, 0) + 1)
            .apply()
    }

    /**
     * Compute skill score for a single puzzle solve.
     * Based on: time vs target, hints used, difficulty level.
     */
    private fun computeSkillScore(gridSize: Int, timeSeconds: Long, hintsUsed: Int, difficulty: Int): Float {
        val targetTime = TARGET_TIMES[gridSize] ?: 300L

        // Time factor: 1.0 if faster than target, decreases as slower
        val timeFactor = (targetTime.toFloat() / max(timeSeconds, 1).toFloat()).coerceIn(0.2f, 2.0f)

        // Hint penalty: each hint reduces score
        val hintPenalty = hintsUsed * 0.1f

        // Difficulty bonus: harder puzzles give more credit
        val difficultyBonus = difficulty * 0.1f

        return ((timeFactor - hintPenalty + difficultyBonus) / 2.0f).coerceIn(0.0f, 1.0f)
    }

    /**
     * Get recommended difficulty for adaptive mode.
     * @param gridSize The intended grid size
     * @return Recommended difficulty (0-4)
     */
    @Suppress("UNUSED_PARAMETER")  // gridSize reserved for future size-specific recommendations
    fun getRecommendedDifficulty(gridSize: Int): Int {
        val stats = getStats()
        val skillScore = stats.skillScore

        // Map skill score to difficulty
        return when {
            skillScore < 0.25f -> 0  // Easy
            skillScore < 0.45f -> 1  // Normal
            skillScore < 0.65f -> 2  // Hard
            skillScore < 0.85f -> 3  // Insane
            else -> 4                // Ludicrous
        }
    }

    fun clearStats() {
        prefs.edit().clear().apply()
    }
}
