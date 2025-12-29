/*
 * StoryManager.kt: Stub for Classik flavor (no Story Mode)
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 *
 * Classik flavor: Story Mode is not available.
 * This stub provides the data classes and a no-op manager for compile compatibility.
 */

package org.yegie.keenkenning.data

import android.content.Context

/**
 * A story book containing multiple chapters.
 */
data class StoryBook(
    val id: String,
    val title: String,
    val subtitle: String,
    val chapters: List<StoryChapter>,
    val theme: BookTheme
)

/**
 * Visual theme for a book.
 */
data class BookTheme(
    val primaryColor: Long,
    val accentColor: Long,
    val coverImageName: String
)

/**
 * A story chapter containing themed puzzles and narrative.
 */
data class StoryChapter(
    val id: String,
    val title: String,
    val description: String,
    val puzzleCount: Int,
    val gridSize: Int,
    val difficulty: Int,
    val theme: StoryTheme,
    val introNarrative: String,
    val outroNarrative: String,
    val bookId: String = "",
    val gameMode: GameMode = GameMode.STANDARD
)

/**
 * Visual theme for a story chapter.
 */
data class StoryTheme(
    val primaryColor: Long,
    val backgroundColor: Long,
    val cellColor: Long,
    val textColor: Long,
    val borderColor: Long,
    val iconName: String
)

/**
 * Player's progress in Story Mode.
 */
data class StoryProgress(
    val currentChapterId: String?,
    val completedChapters: Set<String>,
    val puzzlesCompleted: Map<String, Int>
)

/**
 * Stub StoryManager for Classik flavor.
 * Story Mode is not available in Classik - this provides compile compatibility.
 */
@Suppress("UNUSED_PARAMETER")
class StoryManager(private val context: Context) {

    fun getAllBooks(): List<StoryBook> = emptyList()

    fun getBook(bookId: String): StoryBook? = null

    fun getCurrentChapter(): StoryChapter? = null

    fun getChapter(chapterId: String): StoryChapter? = null

    fun getAllChapters(): List<StoryChapter> = emptyList()

    fun getProgress(): StoryProgress = StoryProgress(null, emptySet(), emptyMap())

    fun isChapterUnlocked(chapterId: String): Boolean = false

    fun markChapterComplete(chapterId: String) { /* no-op */ }

    fun recordPuzzleComplete(chapterId: String) { /* no-op */ }

    fun recordPuzzleCompleted(chapterId: String) { /* no-op */ }

    fun setCurrentChapter(chapterId: String?) { /* no-op */ }

    fun resetProgress() { /* no-op */ }

    suspend fun getIntroNarrative(chapter: StoryChapter, useDynamic: Boolean = false): String =
        chapter.introNarrative

    suspend fun getOutroNarrative(
        chapter: StoryChapter,
        success: Boolean,
        timeSeconds: Int,
        useDynamic: Boolean = false
    ): String = chapter.outroNarrative
}
