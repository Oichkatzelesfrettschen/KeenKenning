/*
 * ClassikStoryManager.kt: Story mode stub for Classik flavor
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning.data

import android.content.Context

/**
 * Stub StoryManager for Classik flavor.
 * Story Mode is not available in Classik - this provides compile compatibility.
 */
@Suppress("UNUSED_PARAMETER")
class ClassikStoryManager(private val context: Context) : StoryManager {
    override fun getAllBooks(): List<StoryBook> = emptyList()

    override fun getBook(bookId: String): StoryBook? = null

    override fun getCurrentBook(): StoryBook? = null

    override fun getAllChapters(): List<StoryChapter> = emptyList()

    override fun getChapter(chapterId: String): StoryChapter? = null

    override fun getProgress(): StoryProgress = StoryProgress(null, emptySet(), emptyMap())

    override fun isChapterUnlocked(chapterId: String): Boolean = false

    override fun setCurrentChapter(chapterId: String?) { /* no-op */ }

    override fun recordPuzzleCompleted(chapterId: String) { /* no-op */ }

    override fun resetProgress() { /* no-op */ }

    override fun setCurrentBook(bookId: String) { /* no-op */ }

    override fun isBookUnlocked(bookId: String): Boolean = false

    override fun getBookProgress(bookId: String): Float = 0f

    override fun getTotalProgress(): Float = 0f

    override suspend fun getIntroNarrative(
        chapter: StoryChapter,
        useDynamic: Boolean
    ): String = chapter.introNarrative

    override suspend fun getOutroNarrative(
        chapter: StoryChapter,
        success: Boolean,
        timeSeconds: Int,
        useDynamic: Boolean
    ): String = chapter.outroNarrative

    override fun releaseNarrativeGenerator() { /* no-op */ }
}
