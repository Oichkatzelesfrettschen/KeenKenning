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
    val implemented: Boolean = false
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
        implemented = true  // Phase 1: UI-only, hide operation symbols
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
        implemented = true  // Clue values use mod N; solver verification WIP
    ),
    KILLER(
        displayName = "Killer",
        description = "No repeated digits in cages",
        iconName = "block",
        cFlags = 0x40,
        phase = 3,
        implemented = true
    ),

    // Phase 4: Research-Backed Innovations
    HINT_MODE(
        displayName = "Tutorial",
        description = "Explainable hints with reasoning",
        iconName = "school",
        cFlags = 0x80,
        phase = 4,
        implemented = true
    ),
    ADAPTIVE(
        displayName = "Adaptive",
        description = "Difficulty adjusts to your skill",
        iconName = "trending_up",
        cFlags = 0x100,
        phase = 4,
        implemented = true
    ),
    STORY(
        displayName = "Story",
        description = "Themed puzzles with narrative",
        iconName = "auto_stories",
        cFlags = 0x200,
        phase = 4,
        implemented = true
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
 */
enum class GridSize(val size: Int, val displayName: String, val usesHex: Boolean = false) {
    SIZE_3(3, "3×3"),
    SIZE_4(4, "4×4"),
    SIZE_5(5, "5×5"),
    SIZE_6(6, "6×6"),
    SIZE_7(7, "7×7"),
    SIZE_8(8, "8×8"),
    SIZE_9(9, "9×9"),
    SIZE_10(10, "10×10", usesHex = true),
    SIZE_12(12, "12×12", usesHex = true),
    SIZE_16(16, "16×16", usesHex = true);

    companion object {
        fun fromInt(size: Int): GridSize = entries.find { it.size == size } ?: SIZE_5
        fun standardSizes(): List<GridSize> = entries.filter { !it.usesHex }
        fun extendedSizes(): List<GridSize> = entries.filter { it.usesHex }
        fun allSizes(): List<GridSize> = entries.toList()
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
