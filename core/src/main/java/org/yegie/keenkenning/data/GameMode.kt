/*
 * GameMode.kt: Game mode definitions and configuration
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning.data

/**
 * Game modes available in Keen Kenning.
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
    BITWISE(
        displayName = "Bitwise",
        description = "XOR operations (high ambiguity)",
        iconName = "memory",
        cFlags = 0x800,  // MODE_BITWISE from keen_modes.h
        phase = 2,
        implemented = true,
        extendedTip = "Introduces XOR (exclusive-or) operations. " +
            "XOR has very high ambiguity: many digit combinations produce the same result. " +
            "Tip: XOR is self-inverse (a ⊕ a = 0) and the identity is 0. " +
            "Best for Hard+ difficulties where ambiguity adds challenge."
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
    ),

    // Visual Theme Mode
    RETRO_8BIT(
        displayName = "Standard 8-Bit",
        description = "Retro pixel art style with voxel buttons",
        iconName = "videogame_asset",
        cFlags = 0x400,
        phase = 1,
        implemented = true,
        extendedTip = "Experience the puzzle in classic 8-bit style! " +
            "Features pixel art graphics, chunky voxel-style buttons, " +
            "and the Press Start 2P font for authentic retro vibes."
    );

    companion object {
        /**
         * Modes available in Classik (legacy Keen).
         * Only Standard and Standard 8-Bit - the classic experience.
         */
        private val CLASSIK_MODES = setOf(STANDARD, RETRO_8BIT)

        private fun flavorConfig(): FlavorConfig = FlavorConfigProvider.get()

        /**
         * Get all modes available for selection in this flavor.
         * Classik: Standard + Standard 8-Bit only (legacy experience)
         * Kenning: All implemented modes (full feature set)
         */
        fun availableModes(): List<GameMode> = entries.filter {
            it.implemented && (flavorConfig().mlEnabled || it in CLASSIK_MODES)
        }

        /**
         * Get all modes including upcoming ones (respects flavor)
         */
        fun allModes(): List<GameMode> = entries.filter {
            flavorConfig().mlEnabled || it in CLASSIK_MODES
        }

        /**
         * Get modes by phase (respects flavor restrictions)
         */
        fun byPhase(phase: Int): List<GameMode> = entries.filter {
            it.phase == phase && (flavorConfig().mlEnabled || it in CLASSIK_MODES)
        }

        /**
         * Check if a specific mode is available in this flavor
         */
        fun isAvailable(mode: GameMode): Boolean =
            mode.implemented && (flavorConfig().mlEnabled || mode in CLASSIK_MODES)

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
        /** Filter sizes by flavor limits */
        private fun inRange(gs: GridSize): Boolean {
            val config = FlavorConfigProvider.get()
            return gs.size in config.minGridSize..config.maxGridSize
        }

        fun fromInt(size: Int): GridSize = entries.find { it.size == size } ?: SIZE_5
        fun standardSizes(): List<GridSize> = entries.filter { !it.usesHex && inRange(it) }
        fun extendedSizes(): List<GridSize> = entries.filter { it.usesHex && inRange(it) }
        fun allSizes(): List<GridSize> = entries.filter { inRange(it) }
        fun stableSizes(): List<GridSize> = entries.filter { it.stability == Stability.STABLE && inRange(it) }

        /** Maximum size that works with extended modes (Zero/Negative/Modular/Powers) */
        const val MAX_EXTENDED_MODE_SIZE = 9
    }
}

/**
 * Difficulty levels with human-readable names.
 *
 * Matches the 7-level system in keen.c DIFFLIST:
 *   0=Easy, 1=Normal, 2=Hard, 3=Extreme, 4=Unreasonable, 5=Ludicrous, 6=Incomprehensible
 *
 * Difficulty progression:
 *   - EASY: Basic single-candidate deductions
 *   - NORMAL: Set deductions (pointing pairs, box/line reduction)
 *   - HARD: Advanced sets (naked/hidden pairs, X-wing patterns)
 *   - EXTREME: Forcing chains, region-based elimination
 *   - UNREASONABLE: Requires some trial-and-error (limited backtracking)
 *   - LUDICROUS: Extensive trial-and-error, deep recursion
 *   - INCOMPREHENSIBLE: Maximum difficulty, may require exceptional solving skills
 */
enum class Difficulty(val level: Int, val displayName: String) {
    EASY(0, "Easy"),
    NORMAL(1, "Normal"),
    HARD(2, "Hard"),
    EXTREME(3, "Extreme"),
    UNREASONABLE(4, "Unreasonable"),
    LUDICROUS(5, "Ludicrous"),
    INCOMPREHENSIBLE(6, "Incomprehensible");

    companion object {
        fun fromInt(level: Int): Difficulty = entries.find { it.level == level } ?: NORMAL
        val DEFAULT = NORMAL

        /** Maximum difficulty available for Classik flavor (3x3-9x9 grids) */
        val CLASSIK_MAX = LUDICROUS

        /** Maximum difficulty available for Kenning flavor (3x3-16x16 grids) */
        val KENNING_MAX = INCOMPREHENSIBLE
    }
}
