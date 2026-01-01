/*
 * StoryManager.kt: Story mode service interface
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning.data

interface StoryManager {
    fun getAllBooks(): List<StoryBook>

    fun getBook(bookId: String): StoryBook?

    fun getCurrentBook(): StoryBook?

    fun getAllChapters(): List<StoryChapter>

    fun getChapter(chapterId: String): StoryChapter?

    fun getProgress(): StoryProgress

    fun isChapterUnlocked(chapterId: String): Boolean

    fun setCurrentChapter(chapterId: String?)

    fun recordPuzzleCompleted(chapterId: String)

    fun resetProgress()

    fun setCurrentBook(bookId: String)

    fun isBookUnlocked(bookId: String): Boolean

    fun getBookProgress(bookId: String): Float

    fun getTotalProgress(): Float

    suspend fun getIntroNarrative(chapter: StoryChapter, useDynamic: Boolean = false): String

    suspend fun getOutroNarrative(
        chapter: StoryChapter,
        success: Boolean,
        timeSeconds: Int,
        useDynamic: Boolean = false
    ): String

    fun releaseNarrativeGenerator()
}
