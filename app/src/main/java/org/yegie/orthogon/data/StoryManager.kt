/*
 * StoryManager.kt: Story Mode chapter and progress management (Phase 4c)
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 */

package org.yegie.orthogon.data

import android.content.Context
import android.content.SharedPreferences

/**
 * A story chapter containing themed puzzles and narrative.
 */
data class StoryChapter(
    val id: String,
    val title: String,
    val description: String,
    val puzzleCount: Int,
    val gridSize: Int,          // Default grid size for chapter
    val difficulty: Int,        // Default difficulty (0-4)
    val theme: StoryTheme,
    val introNarrative: String,
    val outroNarrative: String
)

/**
 * Visual theme for a story chapter.
 */
data class StoryTheme(
    val primaryColor: Long,     // Main accent color
    val backgroundColor: Long,  // Grid background
    val cellColor: Long,        // Cell fill
    val textColor: Long,        // Text color
    val borderColor: Long,      // Cage borders
    val iconName: String        // Material icon for chapter
)

/**
 * Player's progress in Story Mode.
 */
data class StoryProgress(
    val currentChapterId: String?,
    val completedChapters: Set<String>,
    val puzzlesCompleted: Map<String, Int>  // chapterId -> puzzles solved
)

/**
 * Manages Story Mode chapters and player progress.
 */
class StoryManager(context: Context) {

    companion object {
        private const val PREFS_NAME = "orthogon_story_progress"
        private const val KEY_CURRENT_CHAPTER = "current_chapter"
        private const val KEY_COMPLETED_PREFIX = "completed_"
        private const val KEY_PUZZLES_PREFIX = "puzzles_"

        // Predefined story chapters
        val CHAPTERS = listOf(
            StoryChapter(
                id = "origins",
                title = "Origins",
                description = "Where it all began - simple puzzles to find your path",
                puzzleCount = 5,
                gridSize = 4,
                difficulty = 0,
                theme = StoryTheme(
                    primaryColor = 0xFF4CAF50,   // Green
                    backgroundColor = 0xFF1B3B1B,
                    cellColor = 0xFF2D4D2D,
                    textColor = 0xFFE8F5E9,
                    borderColor = 0xFF81C784,
                    iconName = "eco"
                ),
                introNarrative = "In the ancient forest of numbers, a traveler seeks wisdom...",
                outroNarrative = "The path becomes clearer. The journey continues..."
            ),
            StoryChapter(
                id = "depths",
                title = "The Depths",
                description = "Venture into the caverns of complexity",
                puzzleCount = 5,
                gridSize = 5,
                difficulty = 1,
                theme = StoryTheme(
                    primaryColor = 0xFF2196F3,   // Blue
                    backgroundColor = 0xFF0D1B2A,
                    cellColor = 0xFF1B2838,
                    textColor = 0xFFE3F2FD,
                    borderColor = 0xFF64B5F6,
                    iconName = "terrain"
                ),
                introNarrative = "Deep beneath the surface, patterns emerge from shadow...",
                outroNarrative = "You've conquered the depths. The summit awaits."
            ),
            StoryChapter(
                id = "summit",
                title = "The Summit",
                description = "The final challenge - master the mountain",
                puzzleCount = 5,
                gridSize = 6,
                difficulty = 2,
                theme = StoryTheme(
                    primaryColor = 0xFFFF9800,   // Orange
                    backgroundColor = 0xFF2A1810,
                    cellColor = 0xFF3D251A,
                    textColor = 0xFFFFF3E0,
                    borderColor = 0xFFFFB74D,
                    iconName = "landscape"
                ),
                introNarrative = "At the peak, clarity meets challenge...",
                outroNarrative = "You have reached enlightenment. A new journey begins."
            )
        )
    }

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun getProgress(): StoryProgress {
        val completedChapters = mutableSetOf<String>()
        val puzzlesCompleted = mutableMapOf<String, Int>()

        for (chapter in CHAPTERS) {
            if (prefs.getBoolean("${KEY_COMPLETED_PREFIX}${chapter.id}", false)) {
                completedChapters.add(chapter.id)
            }
            val puzzles = prefs.getInt("${KEY_PUZZLES_PREFIX}${chapter.id}", 0)
            if (puzzles > 0) {
                puzzlesCompleted[chapter.id] = puzzles
            }
        }

        return StoryProgress(
            currentChapterId = prefs.getString(KEY_CURRENT_CHAPTER, null),
            completedChapters = completedChapters,
            puzzlesCompleted = puzzlesCompleted
        )
    }

    fun getChapter(chapterId: String): StoryChapter? =
        CHAPTERS.find { it.id == chapterId }

    fun setCurrentChapter(chapterId: String) {
        prefs.edit().putString(KEY_CURRENT_CHAPTER, chapterId).apply()
    }

    fun recordPuzzleCompleted(chapterId: String) {
        val chapter = getChapter(chapterId) ?: return
        val currentCount = prefs.getInt("${KEY_PUZZLES_PREFIX}$chapterId", 0)
        val newCount = currentCount + 1

        val editor = prefs.edit()
        editor.putInt("${KEY_PUZZLES_PREFIX}$chapterId", newCount)

        // Check if chapter is now complete
        if (newCount >= chapter.puzzleCount) {
            editor.putBoolean("${KEY_COMPLETED_PREFIX}$chapterId", true)
        }

        editor.apply()
    }

    fun isChapterUnlocked(chapterId: String): Boolean {
        val chapterIndex = CHAPTERS.indexOfFirst { it.id == chapterId }
        if (chapterIndex <= 0) return true  // First chapter always unlocked

        // Previous chapter must be completed
        val prevChapter = CHAPTERS[chapterIndex - 1]
        return prefs.getBoolean("${KEY_COMPLETED_PREFIX}${prevChapter.id}", false)
    }

    fun getNextChapter(): StoryChapter? {
        val progress = getProgress()
        return CHAPTERS.find { !progress.completedChapters.contains(it.id) && isChapterUnlocked(it.id) }
    }

    fun getAllChapters(): List<StoryChapter> = CHAPTERS

    fun resetProgress() {
        prefs.edit().clear().apply()
    }
}
