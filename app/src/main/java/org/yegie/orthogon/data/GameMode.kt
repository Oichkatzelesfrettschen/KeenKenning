/*
 * GameMode.kt: Game mode definitions and configuration
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 */

package org.yegie.orthogon.data

/**
 * Game modes available in Orthogon.
 *
 * Each mode has:
 * - displayName: Short name shown in UI
 * - description: Longer explanation of the mode
 * - iconName: Material icon identifier
 * - cFlags: Bit flags passed to native layer
 * - phase: Implementation phase (1-4)
 */
enum class GameMode(
    val displayName: String,
    val description: String,
    val iconName: String,
    val cFlags: Int,
    val phase: Int,
    val implemented: Boolean = false,
    /** Extended tips shown in UI (null = use description only) */
    val extendedTip: String? = null
) {
    // Phase 1: Core Modes (Low Effort)
    STANDARD(
        displayName = "Standard",
        description = "All operations (+, -, ×, ÷)",
        iconName = "calculate",
        cFlags = 0x00,
        phase = 1,
        implemented = true
    ),
    MULTIPLICATION_ONLY(
        displayName = "Multiply",
        description = "Only × multiplication operations",
        iconName = "close",
        cFlags = 0x01,
        phase = 1,
        implemented = true
    ),
    MYSTERY(
        displayName = "Mystery",
        description = "Operations hidden - deduce them!",
        iconName = "help_outline",
        cFlags = 0x02,
        phase = 1,
        implemented = true,  // Phase 1: UI-only, hide operation symbols
        extendedTip = "The operation symbols (+, -, ×, ÷) are hidden. " +
            "Tip: Use the clue value and cage size to deduce the operation. " +
            "Small values often mean subtraction or division."
    ),
    ZERO_INCLUSIVE(
        displayName = "Zero Mode",
        description = "Numbers 0 to N-1 (no division)",
        iconName = "exposure_zero",
        cFlags = 0x04,
        phase = 1,
        implemented = true
    ),

    // Phase 2: Extended Operations (Medium Effort)
    EXPONENT(
        displayName = "Powers",
        description = "Includes ^ exponent operation",
        iconName = "superscript",
        cFlags = 0x10,
        phase = 2,
        implemented = true
    ),
    NEGATIVE_NUMBERS(
        displayName = "Negative",
        description = "Range -N to +N (excluding 0)",
        iconName = "remove",
        cFlags = 0x08,
        phase = 2,
        implemented = true
    ),

    // Phase 3: Advanced Constraints (High Effort)
    MODULAR(
        displayName = "Modular",
        description = "Wrap-around arithmetic (mod N)",
        iconName = "loop",
        cFlags = 0x20,
        phase = 3,
        implemented = true,  // Clue values use mod N; solver verification WIP
        extendedTip = "Clue values wrap around using modular arithmetic (mod N). " +
            "Example on a 5×5: 4+3=2 because 7 mod 5 = 2. " +
            "Tip: Think of it like a clock where numbers wrap after N."
    ),
    KILLER(
        displayName = "Killer",
        description = "No repeated digits in cages",
        iconName = "block",
        cFlags = 0x40,
        phase = 3,
        implemented = true,
        extendedTip = "Like Killer Sudoku: each cage must contain unique digits. " +
            "Tip: Look for cages that span multiple rows or columns - " +
            "the no-repeat rule adds extra constraints beyond standard Latin square rules."
    ),

    // Phase 4: Research-Backed Innovations
    // These modes are stubs - puzzles generate but lack specialized content
    HINT_MODE(
        displayName = "Tutorial",
        description = "Step-by-step hints (stub: uses standard hints)",
        iconName = "school",
        cFlags = 0x80,
        phase = 4,
        implemented = true  // Stub: falls back to basic hint system
    ),
    ADAPTIVE(
        displayName = "Adaptive",
        description = "Difficulty adapts (stub: fixed difficulty)",
        iconName = "trending_up",
        cFlags = 0x100,
        phase = 4,
        implemented = true  // Stub: no adaptive algorithm yet
    ),
    STORY(
        displayName = "Story",
        description = "Themed puzzles (stub: standard puzzles)",
        iconName = "auto_stories",
        cFlags = 0x200,
        phase = 4,
        implemented = true  // Stub: no story content yet
    );

    companion object {
        /**
         * Get all modes available for selection (implemented or coming soon)
         */
        fun availableModes(): List<GameMode> = entries.filter { it.implemented }

        /**
         * Get all modes including upcoming ones
         */
        fun allModes(): List<GameMode> = entries.toList()

        /**
         * Get modes by phase
         */
        fun byPhase(phase: Int): List<GameMode> = entries.filter { it.phase == phase }

        /**
         * Default mode for new games
         */
        val DEFAULT = STANDARD
    }
}

/**
 * Extended grid size options.
 * Standard sizes 3-9 use decimal digits.
 * Extended sizes 10-16 use hex digits (A-G).
 *
 * Stability levels:
 * - STABLE (3-9): ML model trained, reliable generation, good touch targets
 * - EXPERIMENTAL (10-12): Generation may timeout, smaller touch targets
 * - ADVANCED (16): Very long generation, requires zoom for usability
 */
enum class GridSize(
    val size: Int,
    val displayName: String,
    val usesHex: Boolean = false,
    val stability: Stability = Stability.STABLE
) {
    SIZE_3(3, "3×3"),
    SIZE_4(4, "4×4"),
    SIZE_5(5, "5×5"),
    SIZE_6(6, "6×6"),
    SIZE_7(7, "7×7"),
    SIZE_8(8, "8×8"),
    SIZE_9(9, "9×9"),
    SIZE_10(10, "10×10", usesHex = true, stability = Stability.EXPERIMENTAL),
    SIZE_12(12, "12×12", usesHex = true, stability = Stability.EXPERIMENTAL),
    SIZE_16(16, "16×16", usesHex = true, stability = Stability.ADVANCED);

    enum class Stability {
        STABLE,       // ML-supported, fast generation, good UX
        EXPERIMENTAL, // May timeout, reduced touch targets
        ADVANCED      // Expert only, may require zoom
    }

    companion object {
        fun fromInt(size: Int): GridSize = entries.find { it.size == size } ?: SIZE_5
        fun standardSizes(): List<GridSize> = entries.filter { !it.usesHex }
        fun extendedSizes(): List<GridSize> = entries.filter { it.usesHex }
        fun allSizes(): List<GridSize> = entries.toList()
        fun stableSizes(): List<GridSize> = entries.filter { it.stability == Stability.STABLE }

        /** Maximum size that works with extended modes (Zero/Negative/Modular/Powers) */
        const val MAX_EXTENDED_MODE_SIZE = 9
    }
}

/**
 * Difficulty levels with human-readable names.
 */
enum class Difficulty(val level: Int, val displayName: String) {
    EASY(0, "Easy"),
    NORMAL(1, "Normal"),
    HARD(2, "Hard"),
    INSANE(3, "Insane"),
    LUDICROUS(4, "Ludicrous");

    companion object {
        fun fromInt(level: Int): Difficulty = entries.find { it.level == level } ?: NORMAL
        val DEFAULT = NORMAL
    }
}
