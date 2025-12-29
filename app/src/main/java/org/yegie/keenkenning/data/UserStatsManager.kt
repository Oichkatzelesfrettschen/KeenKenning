/*
 * UserStatsManager.kt: Player performance tracking for Adaptive Mode (Phase 4b)
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning.data

import android.content.Context
import android.content.SharedPreferences
import kotlin.math.max
import kotlin.math.min

/**
 * Player performance statistics for adaptive difficulty.
 * Tracks solve times, success rates, hints, errors, undos, and mode diversity.
 */
data class PlayerStats(
    val totalPuzzlesSolved: Int = 0,
    val totalPuzzlesAbandoned: Int = 0,
    val totalHintsUsed: Int = 0,
    val averageSolveTimeSeconds: Long = 0,
    // Per-size stats: key = gridSize (3-9), value = (solveCount, avgTimeSeconds)
    val sizeStats: Map<Int, SizeStats> = emptyMap(),
    // Computed skill score: 0.0 (beginner) to 1.0 (expert)
    val skillScore: Float = 0.5f,

    // Extended tracking for Adaptive mode (Phase 4b enhancements)
    /** Current consecutive win streak */
    val currentStreak: Int = 0,
    /** Best win streak achieved */
    val bestStreak: Int = 0,
    /** Average error rate (incorrect entries per puzzle) */
    val averageErrorRate: Float = 0f,
    /** Average undo frequency (undos per puzzle) */
    val averageUndoRate: Float = 0f,
    /** Mode diversity: count of different modes played recently */
    val recentModes: List<String> = emptyList(),
    /** Session count (for long-term tracking) */
    val totalSessions: Int = 0,
    /** Last play timestamp (epoch millis) */
    val lastPlayedTimestamp: Long = 0
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
        private const val PREFS_NAME = "keenkenning_user_stats"
        private const val KEY_TOTAL_SOLVED = "total_solved"
        private const val KEY_TOTAL_ABANDONED = "total_abandoned"
        private const val KEY_TOTAL_HINTS = "total_hints"
        private const val KEY_AVG_TIME = "avg_time"
        private const val KEY_SKILL_SCORE = "skill_score"
        private const val KEY_SIZE_STATS_PREFIX = "size_stats_"

        // Extended tracking keys
        private const val KEY_CURRENT_STREAK = "current_streak"
        private const val KEY_BEST_STREAK = "best_streak"
        private const val KEY_AVG_ERROR_RATE = "avg_error_rate"
        private const val KEY_AVG_UNDO_RATE = "avg_undo_rate"
        private const val KEY_RECENT_MODES = "recent_modes"
        private const val KEY_TOTAL_SESSIONS = "total_sessions"
        private const val KEY_LAST_PLAYED = "last_played"

        /** Maximum recent modes to track for diversity scoring */
        private const val MAX_RECENT_MODES = 20

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
        for (size in 3..16) {  // Extended to support 16x16 grids
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

        // Load recent modes as comma-separated string
        val recentModesStr = prefs.getString(KEY_RECENT_MODES, "") ?: ""
        val recentModes = if (recentModesStr.isBlank()) emptyList()
                          else recentModesStr.split(",")

        return PlayerStats(
            totalPuzzlesSolved = prefs.getInt(KEY_TOTAL_SOLVED, 0),
            totalPuzzlesAbandoned = prefs.getInt(KEY_TOTAL_ABANDONED, 0),
            totalHintsUsed = prefs.getInt(KEY_TOTAL_HINTS, 0),
            averageSolveTimeSeconds = prefs.getLong(KEY_AVG_TIME, 0),
            sizeStats = sizeStats,
            skillScore = prefs.getFloat(KEY_SKILL_SCORE, 0.5f),
            currentStreak = prefs.getInt(KEY_CURRENT_STREAK, 0),
            bestStreak = prefs.getInt(KEY_BEST_STREAK, 0),
            averageErrorRate = prefs.getFloat(KEY_AVG_ERROR_RATE, 0f),
            averageUndoRate = prefs.getFloat(KEY_AVG_UNDO_RATE, 0f),
            recentModes = recentModes,
            totalSessions = prefs.getInt(KEY_TOTAL_SESSIONS, 0),
            lastPlayedTimestamp = prefs.getLong(KEY_LAST_PLAYED, 0)
        )
    }

    /**
     * Record a completed puzzle for stats tracking.
     * @param gridSize The puzzle grid size (3-16)
     * @param solveTimeSeconds Time taken to solve
     * @param hintsUsed Number of hints used
     * @param difficulty Original difficulty level (0-4)
     * @param errorCount Number of incorrect entries made (optional, default 0)
     * @param undoCount Number of undo operations used (optional, default 0)
     * @param modeName Name of the game mode played (optional)
     */
    fun recordPuzzleSolved(
        gridSize: Int,
        solveTimeSeconds: Long,
        hintsUsed: Int,
        difficulty: Int,
        errorCount: Int = 0,
        undoCount: Int = 0,
        modeName: String? = null
    ) {
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

        // Update win streak
        val currentStreak = prefs.getInt(KEY_CURRENT_STREAK, 0) + 1
        val bestStreak = max(prefs.getInt(KEY_BEST_STREAK, 0), currentStreak)
        editor.putInt(KEY_CURRENT_STREAK, currentStreak)
        editor.putInt(KEY_BEST_STREAK, bestStreak)

        // Update error rate (exponential moving average)
        val cellCount = gridSize * gridSize
        val errorRate = errorCount.toFloat() / cellCount
        val oldErrorRate = prefs.getFloat(KEY_AVG_ERROR_RATE, 0f)
        val newErrorRate = if (newTotal == 1) errorRate
                           else oldErrorRate * 0.8f + errorRate * 0.2f
        editor.putFloat(KEY_AVG_ERROR_RATE, newErrorRate)

        // Update undo rate (exponential moving average)
        val undoRate = undoCount.toFloat() / cellCount
        val oldUndoRate = prefs.getFloat(KEY_AVG_UNDO_RATE, 0f)
        val newUndoRate = if (newTotal == 1) undoRate
                          else oldUndoRate * 0.8f + undoRate * 0.2f
        editor.putFloat(KEY_AVG_UNDO_RATE, newUndoRate)

        // Update mode diversity
        modeName?.let { mode ->
            val recentModesStr = prefs.getString(KEY_RECENT_MODES, "") ?: ""
            val recentModes = if (recentModesStr.isBlank()) mutableListOf()
                              else recentModesStr.split(",").toMutableList()
            recentModes.add(0, mode)
            // Keep only the most recent modes
            val trimmed = recentModes.take(MAX_RECENT_MODES)
            editor.putString(KEY_RECENT_MODES, trimmed.joinToString(","))
        }

        // Update last played timestamp
        editor.putLong(KEY_LAST_PLAYED, System.currentTimeMillis())

        // Recompute skill score with enhanced factors
        val skillScore = computeSkillScore(gridSize, solveTimeSeconds, hintsUsed, difficulty, errorCount, undoCount)
        val oldSkill = prefs.getFloat(KEY_SKILL_SCORE, 0.5f)
        // Exponential moving average: 20% new, 80% old for stability
        val newSkill = oldSkill * 0.8f + skillScore * 0.2f
        editor.putFloat(KEY_SKILL_SCORE, newSkill.coerceIn(0.1f, 0.95f))

        editor.apply()
    }

    fun recordPuzzleAbandoned() {
        prefs.edit()
            .putInt(KEY_TOTAL_ABANDONED, prefs.getInt(KEY_TOTAL_ABANDONED, 0) + 1)
            .putInt(KEY_CURRENT_STREAK, 0)  // Reset streak on abandon
            .apply()
    }

    /**
     * Record the start of a new session (app opened).
     */
    fun recordSessionStart() {
        prefs.edit()
            .putInt(KEY_TOTAL_SESSIONS, prefs.getInt(KEY_TOTAL_SESSIONS, 0) + 1)
            .apply()
    }

    /**
     * Get mode diversity score (0.0-1.0).
     * Higher score means more varied mode selection.
     */
    fun getModeDiversityScore(): Float {
        val recentModesStr = prefs.getString(KEY_RECENT_MODES, "") ?: ""
        if (recentModesStr.isBlank()) return 0.5f

        val recentModes = recentModesStr.split(",")
        val uniqueModes = recentModes.toSet().size
        val totalModes = recentModes.size

        // Diversity = unique modes / total modes played
        return if (totalModes == 0) 0.5f
               else (uniqueModes.toFloat() / totalModes).coerceIn(0f, 1f)
    }

    /**
     * Compute skill score for a single puzzle solve.
     *
     * Factors considered (weighted):
     * - Time vs target (40%): faster = higher score
     * - Hint penalty (15%): each hint reduces score
     * - Error penalty (15%): high error rate reduces score
     * - Undo penalty (10%): excessive undos reduce score
     * - Difficulty bonus (20%): harder puzzles give more credit
     */
    private fun computeSkillScore(
        gridSize: Int,
        timeSeconds: Long,
        hintsUsed: Int,
        difficulty: Int,
        errorCount: Int = 0,
        undoCount: Int = 0
    ): Float {
        // Extend target times for larger grids
        val extendedTargetTimes = TARGET_TIMES + mapOf(
            10 to 720L,
            11 to 840L,
            12 to 960L,
            13 to 1080L,
            14 to 1200L,
            15 to 1320L,
            16 to 1500L
        )
        val targetTime = extendedTargetTimes[gridSize] ?: 300L

        // Time factor (40%): 1.0 if faster than target, decreases as slower
        val timeFactor = (targetTime.toFloat() / max(timeSeconds, 1).toFloat()).coerceIn(0.2f, 2.0f)

        // Hint penalty (15%): each hint reduces score
        val hintPenalty = min(hintsUsed * 0.15f, 0.5f)

        // Error penalty (15%): based on error rate relative to grid size
        val cellCount = gridSize * gridSize
        val errorRate = errorCount.toFloat() / cellCount
        val errorPenalty = min(errorRate * 2f, 0.4f)

        // Undo penalty (10%): excessive undos indicate struggle
        val undoRate = undoCount.toFloat() / cellCount
        val undoPenalty = min(undoRate * 1.5f, 0.3f)

        // Difficulty bonus (20%): harder puzzles give more credit
        val difficultyBonus = difficulty * 0.2f

        // Weighted combination
        val rawScore = (timeFactor * 0.4f) +
                       (difficultyBonus * 0.2f) -
                       (hintPenalty * 0.15f) -
                       (errorPenalty * 0.15f) -
                       (undoPenalty * 0.1f)

        return rawScore.coerceIn(0.0f, 1.0f)
    }

    /**
     * Get recommended difficulty for adaptive mode.
     *
     * Uses multiple factors:
     * - Base skill score (from solve history)
     * - Win streak modifier (hot streak = bump up)
     * - Error/undo rates (struggling = bump down)
     * - Mode diversity bonus (varied play = slight bump up)
     *
     * @param gridSize The intended grid size
     * @return Recommended difficulty (0-4)
     */
    fun getRecommendedDifficulty(gridSize: Int): Int {
        val stats = getStats()
        var adjustedScore = stats.skillScore

        // Win streak modifier: +0.05 per win up to +0.15
        val streakBonus = min(stats.currentStreak * 0.05f, 0.15f)
        adjustedScore += streakBonus

        // Error rate penalty: high error rate suggests struggle
        if (stats.averageErrorRate > 0.2f) {
            adjustedScore -= 0.1f
        }

        // Undo rate penalty: excessive undos suggest difficulty
        if (stats.averageUndoRate > 0.3f) {
            adjustedScore -= 0.05f
        }

        // Mode diversity bonus: varied play indicates engagement
        val diversityScore = getModeDiversityScore()
        if (diversityScore > 0.6f) {
            adjustedScore += 0.05f
        }

        // Size-specific adjustment: larger grids are inherently harder
        val sizeAdjustment = when {
            gridSize >= 12 -> -0.1f  // Large grids: recommend slightly easier
            gridSize <= 4 -> 0.1f    // Small grids: can handle harder
            else -> 0f
        }
        adjustedScore += sizeAdjustment

        // Map adjusted score to difficulty
        adjustedScore = adjustedScore.coerceIn(0f, 1f)
        return when {
            adjustedScore < 0.25f -> 0  // Easy
            adjustedScore < 0.45f -> 1  // Normal
            adjustedScore < 0.65f -> 2  // Hard
            adjustedScore < 0.85f -> 3  // Insane
            else -> 4                   // Ludicrous
        }
    }

    /**
     * Get a recommended grid size based on player experience.
     * @return Recommended grid size (3-9 for beginners, up to 16 for experts)
     */
    fun getRecommendedGridSize(): Int {
        val stats = getStats()
        val totalSolved = stats.totalPuzzlesSolved

        return when {
            totalSolved < 5 -> 4      // Beginner: start small
            totalSolved < 15 -> 5     // Learning
            totalSolved < 30 -> 6     // Intermediate
            totalSolved < 60 -> 7     // Experienced
            stats.skillScore > 0.7f -> 8  // Expert: allow larger
            stats.skillScore > 0.85f -> 9 // Master
            else -> 6                 // Default to medium
        }
    }

    fun clearStats() {
        prefs.edit().clear().apply()
    }
}
