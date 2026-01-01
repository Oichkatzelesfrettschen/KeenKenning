/*
 * StoryActivity.kt: Activity hosting the Story Mode book experience
 *
 * Manages the 4-book adventure with page-flip navigation.
 * Launches KeenActivity for individual chapter puzzles.
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.ActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import kotlinx.coroutines.launch
import org.yegie.keenkenning.data.StoryChapter
import org.yegie.keenkenning.data.StoryManager
import org.yegie.keenkenning.ui.StoryBookView
import org.yegie.keenkenning.ui.theme.GameTheme

/**
 * Activity for Story Mode - displays books with page-flip animation.
 */
class StoryActivity : ComponentActivity() {

    companion object {
        /** Intent extra: Book ID to open (default: first unlocked book) */
        const val EXTRA_BOOK_ID = "bookId"

        /** Intent extra: Chapter ID to show (optional, for returning from puzzle) */
        const val EXTRA_CHAPTER_ID = "chapterId"

        /** Intent extra: Whether to show outro (after puzzle completion) */
        const val EXTRA_SHOW_OUTRO = "showOutro"
    }

    private lateinit var storyManager: StoryManager
    private var currentChapterId: String? = null

    /**
     * Activity Result launcher for puzzle completion.
     * Replaces deprecated startActivityForResult/onActivityResult pattern.
     */
    private val puzzleLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result: ActivityResult ->
        handlePuzzleResult(result)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        storyManager = FlavorServices.storyManager(this)

        if (storyManager.getAllBooks().isEmpty()) {
            StoryModeUnavailable.finish(this)
            return
        }

        // Get initial state from intent
        val initialBookId = intent.getStringExtra(EXTRA_BOOK_ID)
        val initialChapterId = intent.getStringExtra(EXTRA_CHAPTER_ID)
        val initialShowOutro = intent.getBooleanExtra(EXTRA_SHOW_OUTRO, false)

        setContent {
            GameTheme {
                // Determine which book to show
                val books = remember { storyManager.getAllBooks() }
                val book = remember(initialBookId, books) {
                    initialBookId?.let { storyManager.getBook(it) }
                        ?: storyManager.getCurrentBook()
                        ?: books.firstOrNull()
                }

                if (book == null) {
                    // No books available (shouldn't happen)
                    finish()
                    return@GameTheme
                }

                val scope = rememberCoroutineScope()

                // Track chapter state
                var chapterIndex by remember {
                    val idx = if (initialChapterId != null) {
                        book.chapters.indexOfFirst { it.id == initialChapterId }
                    } else {
                        -1  // Start at cover
                    }
                    mutableIntStateOf(idx)
                }

                var showOutro by remember { mutableStateOf(initialShowOutro) }

                // Dynamic narrative state
                var dynamicNarratives by remember {
                    mutableStateOf<Map<String, Pair<String?, String?>>>(emptyMap())
                }
                var isLoadingNarrative by remember { mutableStateOf(false) }

                // Generate narrative when chapter changes
                LaunchedEffect(chapterIndex, showOutro) {
                    if (chapterIndex < 0 || chapterIndex >= book.chapters.size) return@LaunchedEffect
                    val chapter = book.chapters[chapterIndex]

                    // Only generate if not already cached
                    val existing = dynamicNarratives[chapter.id]
                    val needsIntro = !showOutro && existing?.first == null
                    val needsOutro = showOutro && existing?.second == null

                    if (needsIntro || needsOutro) {
                        isLoadingNarrative = true
                        scope.launch {
                            try {
                                val intro = if (needsIntro) {
                                    storyManager.getIntroNarrative(chapter, useDynamic = true)
                                } else {
                                    existing?.first
                                }
                                val outro = if (needsOutro) {
                                    storyManager.getOutroNarrative(
                                        chapter,
                                        success = true,
                                        timeSeconds = 0,
                                        useDynamic = true
                                    )
                                } else {
                                    existing?.second
                                }
                                dynamicNarratives = dynamicNarratives + (chapter.id to Pair(intro, outro))
                            } finally {
                                isLoadingNarrative = false
                            }
                        }
                    }
                }

                StoryBookView(
                    book = book,
                    currentChapterIndex = chapterIndex,
                    showOutro = showOutro,
                    dynamicNarratives = dynamicNarratives,
                    isLoadingNarrative = isLoadingNarrative,
                    onStartChapter = { chapter ->
                        currentChapterId = chapter.id
                        launchPuzzle(chapter)
                    },
                    onNextChapter = {
                        // Navigate forward in the book
                        if (showOutro) {
                            // Move to next chapter intro
                            showOutro = false
                            if (chapterIndex < book.chapters.size - 1) {
                                chapterIndex++
                            }
                        }
                    },
                    onPreviousChapter = {
                        // Navigate backward
                        if (!showOutro && chapterIndex >= 0) {
                            if (chapterIndex == 0) {
                                chapterIndex = -1  // Back to cover
                            } else {
                                chapterIndex--
                                // Show previous chapter's outro when going back
                                showOutro = true
                            }
                        } else if (showOutro) {
                            showOutro = false  // Back to same chapter's intro
                        }
                    },
                    onClose = {
                        finish()
                    },
                    modifier = Modifier.fillMaxSize().testTag("storyBookRoot")
                )
            }
        }
    }

    /**
     * Launch the puzzle activity for a specific chapter.
     */
    private fun launchPuzzle(chapter: StoryChapter) {
        storyManager.setCurrentChapter(chapter.id)

        val intent = Intent(this, KeenActivity::class.java).apply {
            putExtra(MenuActivity.GAME_CONT, false)
            putExtra(MenuActivity.GAME_SIZE, chapter.gridSize)
            putExtra(MenuActivity.GAME_DIFF, chapter.difficulty)
            putExtra(MenuActivity.GAME_MODE, chapter.gameMode.name)
            putExtra(MenuActivity.GAME_SEED, System.currentTimeMillis())
            // Story mode extras
            putExtra("storyChapterId", chapter.id)
            putExtra("storyBookId", chapter.bookId)
            putExtra("storyPuzzleCount", chapter.puzzleCount)
        }
        puzzleLauncher.launch(intent)
    }

    /**
     * Handle puzzle completion result from Activity Result API.
     */
    private fun handlePuzzleResult(result: ActivityResult) {
        val chapterId = currentChapterId ?: return
        val chapter = storyManager.getChapter(chapterId) ?: return
        val book = storyManager.getBook(chapter.bookId) ?: return

        // Check if puzzle was completed
        val puzzleCompleted = result.data?.getBooleanExtra("puzzleCompleted", false) ?: false
        if (puzzleCompleted) {
            storyManager.recordPuzzleCompleted(chapterId)
        }

        // Restart with updated state
        val newIntent = Intent(this, StoryActivity::class.java).apply {
            putExtra(EXTRA_BOOK_ID, book.id)
            putExtra(EXTRA_CHAPTER_ID, chapterId)
            // Show outro if chapter is now complete
            val progress = storyManager.getProgress()
            putExtra(EXTRA_SHOW_OUTRO, progress.completedChapters.contains(chapterId))
        }
        finish()
        startActivity(newIntent)
    }

    override fun onDestroy() {
        if (::storyManager.isInitialized) {
            storyManager.releaseNarrativeGenerator()
        }
        super.onDestroy()
    }
}
