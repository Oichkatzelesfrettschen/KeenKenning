/*
 * StoryManager.kt: Story Mode chapter and progress management (Phase 4c)
 *
 * Four-book adventure structure:
 * - Book I: The Awakening (3x3-4x4) - Introduction to number puzzles
 * - Book II: The Journey (5x5-6x6) - Expanding horizons
 * - Book III: The Trials (7x7-8x8) - Advanced challenges
 * - Book IV: The Mastery (9x9) - Ultimate puzzles with all game modes
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 */

package org.yegie.orthogon.data

import android.content.Context
import android.content.SharedPreferences

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
 * Visual theme for a book (applied to all its chapters).
 */
data class BookTheme(
    val primaryColor: Long,
    val accentColor: Long,
    val coverImageName: String  // Resource name for book cover illustration
)

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
    val outroNarrative: String,
    val bookId: String = "",    // Parent book reference
    val gameMode: GameMode = GameMode.STANDARD  // Game mode for this chapter
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
class StoryManager(private val context: Context) {

    companion object {
        private const val PREFS_NAME = "orthogon_story_progress"
        private const val KEY_CURRENT_CHAPTER = "current_chapter"
        private const val KEY_CURRENT_BOOK = "current_book"
        private const val KEY_COMPLETED_PREFIX = "completed_"
        private const val KEY_PUZZLES_PREFIX = "puzzles_"

        // =================================================================
        // BOOK I: THE AWAKENING (3x3-4x4 grids, Easy difficulty)
        // Theme: A traveler discovers an ancient numerical garden
        // =================================================================
        private val BOOK_I_CHAPTERS = listOf(
            StoryChapter(
                id = "b1_c1_first_steps",
                title = "First Steps",
                description = "The garden gate opens",
                puzzleCount = 3,
                gridSize = 3,
                difficulty = 0,
                theme = StoryTheme(
                    primaryColor = 0xFF66BB6A,   // Light green
                    backgroundColor = 0xFF1B3B1B,
                    cellColor = 0xFF2D4D2D,
                    textColor = 0xFFE8F5E9,
                    borderColor = 0xFF81C784,
                    iconName = "spa"
                ),
                introNarrative = """
                    |The ancient gate creaks open, revealing a garden where numbers
                    |bloom like flowers. Each puzzle is a path through the foliage.
                    |Begin with the simplest arrangements - three by three.
                """.trimMargin(),
                outroNarrative = "The first petals fall into place. The garden welcomes you.",
                bookId = "book1"
            ),
            StoryChapter(
                id = "b1_c2_growing_wisdom",
                title = "Growing Wisdom",
                description = "The paths multiply",
                puzzleCount = 4,
                gridSize = 4,
                difficulty = 0,
                theme = StoryTheme(
                    primaryColor = 0xFF4CAF50,   // Green
                    backgroundColor = 0xFF1B3B1B,
                    cellColor = 0xFF2D4D2D,
                    textColor = 0xFFE8F5E9,
                    borderColor = 0xFF66BB6A,
                    iconName = "eco"
                ),
                introNarrative = """
                    |The garden expands. Four by four, the flower beds grow larger.
                    |New operations appear: addition, subtraction, multiplication.
                    |Each cage contains a secret waiting to be solved.
                """.trimMargin(),
                outroNarrative = "You've mastered the growing garden. But whispers speak of deeper paths...",
                bookId = "book1"
            ),
            StoryChapter(
                id = "b1_c3_the_gardeners_test",
                title = "The Gardener's Test",
                description = "Prove your worth",
                puzzleCount = 5,
                gridSize = 4,
                difficulty = 1,
                theme = StoryTheme(
                    primaryColor = 0xFF388E3C,   // Darker green
                    backgroundColor = 0xFF1B3B1B,
                    cellColor = 0xFF2D4D2D,
                    textColor = 0xFFE8F5E9,
                    borderColor = 0xFF4CAF50,
                    iconName = "park"
                ),
                introNarrative = """
                    |The old gardener appears. "Show me what you've learned,"
                    |she says. "Division is the final tool in your kit."
                    |Complete these puzzles to earn passage beyond the garden walls.
                """.trimMargin(),
                outroNarrative = """
                    |The gardener nods with approval. "You are ready for the journey."
                    |She hands you an ancient map. The path leads to distant mountains.
                """.trimMargin(),
                bookId = "book1"
            )
        )

        // =================================================================
        // BOOK II: THE JOURNEY (5x5-6x6 grids, Normal difficulty)
        // Theme: Traversing the wilderness between garden and mountains
        // =================================================================
        private val BOOK_II_CHAPTERS = listOf(
            StoryChapter(
                id = "b2_c1_forest_crossing",
                title = "Forest Crossing",
                description = "Into the wild",
                puzzleCount = 4,
                gridSize = 5,
                difficulty = 1,
                theme = StoryTheme(
                    primaryColor = 0xFF8BC34A,   // Lime green
                    backgroundColor = 0xFF1A2F1A,
                    cellColor = 0xFF2A402A,
                    textColor = 0xFFDCEDC8,
                    borderColor = 0xFF9CCC65,
                    iconName = "forest"
                ),
                introNarrative = """
                    |Beyond the garden walls, a vast forest stretches to the horizon.
                    |The trees whisper puzzles older than memory.
                    |Five rows, five columns - the wilderness demands more.
                """.trimMargin(),
                outroNarrative = "The forest path opens before you. Rivers lie ahead.",
                bookId = "book2"
            ),
            StoryChapter(
                id = "b2_c2_river_crossing",
                title = "River Crossing",
                description = "Bridges of logic",
                puzzleCount = 5,
                gridSize = 5,
                difficulty = 1,
                theme = StoryTheme(
                    primaryColor = 0xFF03A9F4,   // Light blue
                    backgroundColor = 0xFF0D2137,
                    cellColor = 0xFF1A3350,
                    textColor = 0xFFE1F5FE,
                    borderColor = 0xFF4FC3F7,
                    iconName = "water"
                ),
                introNarrative = """
                    |A great river blocks your path. Ancient stone bridges
                    |span the water, but each requires a puzzle to cross.
                    |The current carries whispers of mystery operations...
                """.trimMargin(),
                outroNarrative = "The river is crossed. The mountain foothills beckon.",
                bookId = "book2",
                gameMode = GameMode.MYSTERY
            ),
            StoryChapter(
                id = "b2_c3_mountain_approach",
                title = "Mountain Approach",
                description = "The ascent begins",
                puzzleCount = 5,
                gridSize = 6,
                difficulty = 2,
                theme = StoryTheme(
                    primaryColor = 0xFF795548,   // Brown
                    backgroundColor = 0xFF2C1810,
                    cellColor = 0xFF3D2920,
                    textColor = 0xFFEFEBE9,
                    borderColor = 0xFF8D6E63,
                    iconName = "terrain"
                ),
                introNarrative = """
                    |The ground rises. Six by six grids mark the mountain paths.
                    |Each step higher demands greater precision.
                    |The summit awaits those who persevere.
                """.trimMargin(),
                outroNarrative = "You've reached the mountain's base. The Trials await.",
                bookId = "book2"
            )
        )

        // =================================================================
        // BOOK III: THE TRIALS (7x7-8x8 grids, Hard difficulty)
        // Theme: Overcoming challenges in the sacred mountain temples
        // =================================================================
        private val BOOK_III_CHAPTERS = listOf(
            StoryChapter(
                id = "b3_c1_temple_entrance",
                title = "Temple Entrance",
                description = "The ancient doors",
                puzzleCount = 4,
                gridSize = 7,
                difficulty = 2,
                theme = StoryTheme(
                    primaryColor = 0xFF9C27B0,   // Purple
                    backgroundColor = 0xFF1A0D1A,
                    cellColor = 0xFF2D1A2D,
                    textColor = 0xFFF3E5F5,
                    borderColor = 0xFFBA68C8,
                    iconName = "temple_hindu"
                ),
                introNarrative = """
                    |The mountain temple rises before you. Carved in ancient stone,
                    |puzzles guard every passage. Seven by seven - the sacred number.
                    |Only the worthy may enter.
                """.trimMargin(),
                outroNarrative = "The temple doors open. Deeper mysteries await within.",
                bookId = "book3"
            ),
            StoryChapter(
                id = "b3_c2_hall_of_mirrors",
                title = "Hall of Mirrors",
                description = "Nothing is as it seems",
                puzzleCount = 5,
                gridSize = 7,
                difficulty = 2,
                theme = StoryTheme(
                    primaryColor = 0xFF7B1FA2,   // Deep purple
                    backgroundColor = 0xFF1A0D1A,
                    cellColor = 0xFF2D1A2D,
                    textColor = 0xFFE1BEE7,
                    borderColor = 0xFFAB47BC,
                    iconName = "auto_awesome"
                ),
                introNarrative = """
                    |The Hall of Mirrors reflects possibilities and illusions.
                    |Mystery mode activates - operations are hidden.
                    |Trust your instincts. The patterns will reveal themselves.
                """.trimMargin(),
                outroNarrative = "You've seen through the illusions. The inner sanctum calls.",
                bookId = "book3",
                gameMode = GameMode.MYSTERY
            ),
            StoryChapter(
                id = "b3_c3_inner_sanctum",
                title = "Inner Sanctum",
                description = "The heart of the temple",
                puzzleCount = 5,
                gridSize = 8,
                difficulty = 3,
                theme = StoryTheme(
                    primaryColor = 0xFF673AB7,   // Indigo
                    backgroundColor = 0xFF0D0A1A,
                    cellColor = 0xFF1A152D,
                    textColor = 0xFFD1C4E9,
                    borderColor = 0xFF9575CD,
                    iconName = "lens_blur"
                ),
                introNarrative = """
                    |Eight by eight - the grid of mastery. The sanctum glows with power.
                    |No duplicates may exist within each cage - the Killer constraint.
                    |Focus completely. Victory requires perfection.
                """.trimMargin(),
                outroNarrative = "The sanctum's secrets are yours. One final test remains...",
                bookId = "book3",
                gameMode = GameMode.KILLER
            )
        )

        // =================================================================
        // BOOK IV: THE MASTERY (9x9 grids, Insane difficulty)
        // Theme: Ascending to enlightenment at the mountain peak
        // =================================================================
        private val BOOK_IV_CHAPTERS = listOf(
            StoryChapter(
                id = "b4_c1_final_ascent",
                title = "Final Ascent",
                description = "The path to the peak",
                puzzleCount = 4,
                gridSize = 9,
                difficulty = 3,
                theme = StoryTheme(
                    primaryColor = 0xFFFF9800,   // Orange
                    backgroundColor = 0xFF2A1810,
                    cellColor = 0xFF3D251A,
                    textColor = 0xFFFFF3E0,
                    borderColor = 0xFFFFB74D,
                    iconName = "landscape"
                ),
                introNarrative = """
                    |The summit is near. Nine by nine - the ultimate challenge.
                    |All your skills converge for this final climb.
                    |The mountain tests every lesson you've learned.
                """.trimMargin(),
                outroNarrative = "The peak is within reach. One more step to enlightenment.",
                bookId = "book4"
            ),
            StoryChapter(
                id = "b4_c2_summit",
                title = "The Summit",
                description = "Where numbers touch the sky",
                puzzleCount = 5,
                gridSize = 9,
                difficulty = 3,
                theme = StoryTheme(
                    primaryColor = 0xFFF57C00,   // Deep orange
                    backgroundColor = 0xFF2A1810,
                    cellColor = 0xFF3D251A,
                    textColor = 0xFFFFE0B2,
                    borderColor = 0xFFFF9800,
                    iconName = "filter_hdr"
                ),
                introNarrative = """
                    |You stand at the summit. The world spreads below in infinite patterns.
                    |Modular arithmetic bends the rules - numbers wrap around like the horizon.
                    |Think in cycles. Think in loops.
                """.trimMargin(),
                outroNarrative = "The view from the top changes everything. True mastery awaits.",
                bookId = "book4",
                gameMode = GameMode.MODULAR
            ),
            StoryChapter(
                id = "b4_c3_enlightenment",
                title = "Enlightenment",
                description = "Beyond numbers",
                puzzleCount = 5,
                gridSize = 9,
                difficulty = 4,
                theme = StoryTheme(
                    primaryColor = 0xFFFFD700,   // Gold
                    backgroundColor = 0xFF1A1500,
                    cellColor = 0xFF2D2200,
                    textColor = 0xFFFFFDE7,
                    borderColor = 0xFFFFEB3B,
                    iconName = "wb_sunny"
                ),
                introNarrative = """
                    |At last, enlightenment. The puzzles become meditation.
                    |Every mode, every constraint - you have mastered them all.
                    |These final challenges are your graduation. Your triumph.
                """.trimMargin(),
                outroNarrative = """
                    |The journey is complete. But every ending is a new beginning.
                    |The garden awaits new travelers. Perhaps you will guide them someday.
                    |Thank you for playing Orthogon.
                """.trimMargin(),
                bookId = "book4"
            )
        )

        // All books
        val BOOKS = listOf(
            StoryBook(
                id = "book1",
                title = "Book I",
                subtitle = "The Awakening",
                chapters = BOOK_I_CHAPTERS,
                theme = BookTheme(
                    primaryColor = 0xFF4CAF50,
                    accentColor = 0xFF81C784,
                    coverImageName = "story_book1_cover"
                )
            ),
            StoryBook(
                id = "book2",
                title = "Book II",
                subtitle = "The Journey",
                chapters = BOOK_II_CHAPTERS,
                theme = BookTheme(
                    primaryColor = 0xFF2196F3,
                    accentColor = 0xFF64B5F6,
                    coverImageName = "story_book2_cover"
                )
            ),
            StoryBook(
                id = "book3",
                title = "Book III",
                subtitle = "The Trials",
                chapters = BOOK_III_CHAPTERS,
                theme = BookTheme(
                    primaryColor = 0xFF9C27B0,
                    accentColor = 0xFFBA68C8,
                    coverImageName = "story_book3_cover"
                )
            ),
            StoryBook(
                id = "book4",
                title = "Book IV",
                subtitle = "The Mastery",
                chapters = BOOK_IV_CHAPTERS,
                theme = BookTheme(
                    primaryColor = 0xFFFF9800,
                    accentColor = 0xFFFFB74D,
                    coverImageName = "story_book4_cover"
                )
            )
        )

        // Flat list of all chapters (for backward compatibility)
        val CHAPTERS = BOOKS.flatMap { it.chapters }
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

    fun getAllBooks(): List<StoryBook> = BOOKS

    fun getBook(bookId: String): StoryBook? = BOOKS.find { it.id == bookId }

    fun getCurrentBook(): StoryBook? {
        val bookId = prefs.getString(KEY_CURRENT_BOOK, null)
        return bookId?.let { getBook(it) } ?: BOOKS.firstOrNull()
    }

    fun setCurrentBook(bookId: String) {
        prefs.edit().putString(KEY_CURRENT_BOOK, bookId).apply()
    }

    fun isBookUnlocked(bookId: String): Boolean {
        val bookIndex = BOOKS.indexOfFirst { it.id == bookId }
        if (bookIndex <= 0) return true  // First book always unlocked

        // Previous book must have all chapters completed
        val prevBook = BOOKS[bookIndex - 1]
        return prevBook.chapters.all { chapter ->
            prefs.getBoolean("${KEY_COMPLETED_PREFIX}${chapter.id}", false)
        }
    }

    fun getBookProgress(bookId: String): Float {
        val book = getBook(bookId) ?: return 0f
        val completedChapters = book.chapters.count { chapter ->
            prefs.getBoolean("${KEY_COMPLETED_PREFIX}${chapter.id}", false)
        }
        return completedChapters.toFloat() / book.chapters.size
    }

    fun getTotalProgress(): Float {
        val totalChapters = CHAPTERS.size
        val completedChapters = CHAPTERS.count { chapter ->
            prefs.getBoolean("${KEY_COMPLETED_PREFIX}${chapter.id}", false)
        }
        return completedChapters.toFloat() / totalChapters
    }

    fun resetProgress() {
        prefs.edit().clear().apply()
    }

    // =========================================================================
    // Dynamic Narrative Generation (SmolLM-powered)
    // =========================================================================

    private val narrativeGenerator: NarrativeGenerator by lazy {
        NarrativeGenerator.getInstance(context)
    }

    /**
     * Get the intro narrative for a chapter.
     * Uses AI generation if available, falls back to static narrative.
     *
     * @param chapter The chapter to get intro for
     * @param useDynamic Whether to attempt AI-generated narrative
     * @return Intro narrative text
     */
    suspend fun getIntroNarrative(chapter: StoryChapter, useDynamic: Boolean = true): String {
        if (!useDynamic) {
            return chapter.introNarrative
        }

        return try {
            narrativeGenerator.generateIntro(
                chapterTitle = chapter.title,
                theme = chapter.theme.iconName,
                difficulty = chapter.difficulty
            )
        } catch (e: Exception) {
            chapter.introNarrative
        }
    }

    /**
     * Get the outro narrative for a chapter.
     * Uses AI generation if available, falls back to static narrative.
     *
     * @param chapter The chapter to get outro for
     * @param success Whether the player completed the chapter successfully
     * @param timeSeconds Time taken to complete (for context)
     * @param useDynamic Whether to attempt AI-generated narrative
     * @return Outro narrative text
     */
    suspend fun getOutroNarrative(
        chapter: StoryChapter,
        success: Boolean = true,
        timeSeconds: Int = 0,
        useDynamic: Boolean = true
    ): String {
        if (!useDynamic) {
            return chapter.outroNarrative
        }

        return try {
            narrativeGenerator.generateOutro(
                chapterTitle = chapter.title,
                success = success,
                timeSeconds = timeSeconds
            )
        } catch (e: Exception) {
            chapter.outroNarrative
        }
    }

    /**
     * Release narrative generator resources.
     * Call when story mode is exited.
     */
    fun releaseNarrativeGenerator() {
        narrativeGenerator.close()
    }
}
