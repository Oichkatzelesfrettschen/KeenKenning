/*
 * StoryModels.kt: Story mode shared data types
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning.data

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
