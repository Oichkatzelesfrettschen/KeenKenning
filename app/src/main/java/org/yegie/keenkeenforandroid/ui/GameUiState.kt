/*
 * GameUiState.kt: UI state data classes for the game
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKeen Contributors
 */

package org.yegie.keenkeenforandroid.ui

import org.yegie.keenkeenforandroid.data.GameMode

/**
 * Colorblind accessibility modes.
 * Each mode provides optimized color palettes for specific types of color vision deficiency.
 */
enum class ColorblindMode(val displayName: String, val description: String) {
    NORMAL("Normal", "Standard colors"),
    DEUTERANOPIA("Deuteranopia", "Green-blind friendly"),
    PROTANOPIA("Protanopia", "Red-blind friendly"),
    TRITANOPIA("Tritanopia", "Blue-blind friendly"),
    HIGH_CONTRAST("High Contrast", "Maximum visibility")
}

data class GameUiState(
    val size: Int = 0,
    val cells: List<List<UiCell>> = emptyList(),
    val zones: List<UiZone> = emptyList(),
    val activeCell: Pair<Int, Int>? = null,
    val isSolved: Boolean = false,
    val isInputtingNotes: Boolean = false,
    val showSmartHints: Boolean = false,  // AI-computed probability hints (default off)
    val isLoading: Boolean = false,
    val showInfoDialog: Boolean = false,
    val showVictoryAnimation: Boolean = false,  // Solitaire-style bounce animation
    val victoryAnimationComplete: Boolean = false,
    // Timer and game metadata
    val elapsedTimeSeconds: Long = 0,
    val timerRunning: Boolean = false,
    val difficulty: Int = 0,  // 0=Easy, 1=Normal, 2=Hard, 3=Insane, 4=Ludicrous
    val difficultyName: String = "Easy",
    val isAiGenerated: Boolean = false,
    // Settings
    val showSettingsDialog: Boolean = false,
    val darkTheme: Boolean = true,  // Default dark theme
    val fontScale: Float = 1.0f,   // 1.0 = normal, 1.25 = large, 1.5 = extra large
    val showTimer: Boolean = true,  // Allow hiding timer to reduce anxiety
    val immersiveMode: Boolean = false,  // Full-screen immersive (auto-hide UI)
    val uiVisible: Boolean = true,  // TopBar/BottomBar visibility state
    val colorblindMode: ColorblindMode = ColorblindMode.NORMAL,  // Accessibility: colorblind support
    // Save/Load
    val showSaveDialog: Boolean = false,
    val showLoadDialog: Boolean = false,
    // Game mode (affects display, e.g., MYSTERY hides operations)
    val gameMode: GameMode = GameMode.STANDARD,
    // Hint system (Phase 4a: Tutorial Mode)
    val showHintDialog: Boolean = false,
    val currentHint: HintInfo? = null,
    val hintsUsed: Int = 0  // Track hints for adaptive scoring
)

/**
 * Information about a hint for the current cell.
 * Used in Tutorial/Hint Mode (Phase 4a).
 */
data class HintInfo(
    val suggestedDigit: Int,
    val confidence: Float,  // 0.0-1.0, higher = more certain
    val reasoning: String,  // Human-readable explanation
    val cellX: Int,
    val cellY: Int
)

data class UiCell(
    val x: Int,
    val y: Int,
    val value: Int?,
    val notes: List<Boolean>, 
    val smartHintProbs: List<Float>? = null,  // AI probability hints per digit
    val zoneId: Int,
    val isSelected: Boolean,
    val borders: CellBorders = CellBorders(),
    val clue: String? = null
)

data class CellBorders(
    val top: Boolean = false,
    val bottom: Boolean = false,
    val left: Boolean = false,
    val right: Boolean = false
)

data class UiZone(
    val id: Int,
    val label: String,
    val color: Long
)
