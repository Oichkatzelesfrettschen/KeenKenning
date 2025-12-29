/*
 * AccessibilityUtils.kt: TalkBack and screen reader support utilities
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 *
 * Provides accessibility helpers for:
 * - Live region announcements (timer, game state changes)
 * - Semantic content descriptions
 * - Heading levels for navigation structure
 * - State descriptions for toggles
 * - Screen reader traversal hints
 */

package org.yegie.keenkenning.ui

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import android.view.accessibility.AccessibilityManager
import kotlinx.coroutines.delay

/**
 * Accessibility configuration for the game.
 */
data class AccessibilityConfig(
    /** Whether TalkBack or similar screen reader is enabled */
    val screenReaderEnabled: Boolean,
    /** Whether to announce timer updates (every minute) */
    val announceTimerUpdates: Boolean,
    /** Whether to use polite (non-interrupting) announcements */
    val usePoliteAnnouncements: Boolean
)

/**
 * Detect if a screen reader is currently active.
 */
@Composable
fun rememberAccessibilityConfig(): AccessibilityConfig {
    val context = LocalContext.current
    return remember(context) {
        val am = context.getSystemService(android.content.Context.ACCESSIBILITY_SERVICE)
                as? AccessibilityManager
        AccessibilityConfig(
            screenReaderEnabled = am?.isEnabled == true && am.isTouchExplorationEnabled,
            announceTimerUpdates = true,
            usePoliteAnnouncements = true
        )
    }
}

/**
 * Announce a message to screen readers.
 * Uses View.announceForAccessibility for immediate announcements.
 */
@Composable
fun AnnounceForAccessibility(
    message: String?,
    key: Any? = message
) {
    val view = LocalView.current
    LaunchedEffect(key) {
        if (!message.isNullOrBlank()) {
            // Small delay ensures the announcement happens after layout
            delay(100)
            view.announceForAccessibility(message)
        }
    }
}

/**
 * Modifier extension for live region content (timer, scores, dynamic text).
 * Content changes are announced to screen readers.
 *
 * @param mode Polite = waits for current speech, Assertive = interrupts
 */
fun Modifier.liveRegionAnnouncement(
    mode: LiveRegionMode = LiveRegionMode.Polite
): Modifier = this.semantics {
    liveRegion = mode
}

/**
 * Modifier extension to mark content as a heading for navigation.
 * TalkBack users can jump between headings using swipe gestures.
 */
fun Modifier.accessibilityHeading(): Modifier = this.semantics {
    heading()
}

/**
 * Modifier for toggle buttons with proper state description.
 *
 * @param label The control label (e.g., "Notes mode")
 * @param isOn Current toggle state
 */
fun Modifier.toggleStateDescription(
    label: String,
    isOn: Boolean
): Modifier = this.semantics {
    stateDescription = if (isOn) "$label on" else "$label off"
    contentDescription = "$label, ${if (isOn) "on" else "off"}, double tap to toggle"
}

/**
 * Build accessibility description for a game cell.
 * Provides complete context for screen reader users.
 */
fun buildCellAccessibilityDescription(
    x: Int,
    y: Int,
    value: Int?,
    clue: String?,
    isSelected: Boolean,
    notes: List<Boolean>?,
    puzzleSize: Int
): String {
    val parts = mutableListOf<String>()

    // Position (1-indexed for users)
    parts.add("Row ${y + 1}, column ${x + 1}")

    // Value or empty
    if (value != null && value > 0) {
        parts.add("contains ${formatValueForSpeech(value)}")
    } else {
        parts.add("empty")
    }

    // Active notes
    val activeNotes = notes?.mapIndexedNotNull { idx, set ->
        if (set && idx < puzzleSize) idx + 1 else null
    } ?: emptyList()
    if (activeNotes.isNotEmpty()) {
        parts.add("notes: ${activeNotes.joinToString(", ")}")
    }

    // Clue (cage constraint)
    if (!clue.isNullOrBlank()) {
        parts.add("cage ${formatClueForSpeech(clue)}")
    }

    // Selection state
    if (isSelected) {
        parts.add("selected")
    }

    return parts.joinToString(", ")
}

/**
 * Format cell value for speech (handles hex digits).
 */
private fun formatValueForSpeech(value: Int): String {
    return when {
        value <= 9 -> value.toString()
        value == 10 -> "A (10)"
        value == 11 -> "B (11)"
        value == 12 -> "C (12)"
        value == 13 -> "D (13)"
        value == 14 -> "E (14)"
        value == 15 -> "F (15)"
        value == 16 -> "G (16)"
        else -> value.toString()
    }
}

/**
 * Format cage clue for speech (expands operation symbols).
 */
private fun formatClueForSpeech(clue: String): String {
    return clue
        .replace("+", " plus")
        .replace("-", " minus")
        .replace("×", " times")
        .replace("x", " times")
        .replace("÷", " divided by")
        .replace("/", " divided by")
        .replace("^", " to the power of")
}

/**
 * Format elapsed time for accessibility announcement.
 */
fun formatTimeForSpeech(seconds: Long): String {
    val mins = seconds / 60
    val secs = seconds % 60
    return when {
        mins == 0L -> "$secs seconds"
        secs == 0L -> "$mins ${if (mins == 1L) "minute" else "minutes"}"
        else -> "$mins ${if (mins == 1L) "minute" else "minutes"} and $secs seconds"
    }
}

/**
 * Build victory announcement for screen readers.
 */
fun buildVictoryAnnouncement(
    gridSize: Int,
    difficulty: String,
    elapsedSeconds: Long
): String {
    val timeStr = formatTimeForSpeech(elapsedSeconds)
    return "Congratulations! You solved the $gridSize by $gridSize $difficulty puzzle in $timeStr"
}

/**
 * Modifier to clear default semantics and set custom description.
 * Use for decorative elements that shouldn't be announced individually.
 */
fun Modifier.decorative(): Modifier = this.clearAndSetSemantics { }

/**
 * Modifier for game grid with collection semantics.
 */
fun Modifier.gridSemantics(
    rows: Int,
    columns: Int
): Modifier = this.semantics {
    contentDescription = "$rows by $columns Keen puzzle grid"
}

/**
 * Content description for number input button.
 */
fun numberButtonDescription(number: Int, isNoteMode: Boolean): String {
    val action = if (isNoteMode) "Add note" else "Enter"
    val numStr = formatValueForSpeech(number)
    return "$action $numStr"
}

/**
 * Hint for game controls section.
 */
const val CONTROLS_HINT = "Game controls. Use number buttons to fill cells, " +
        "undo to reverse, clear to erase, notes to toggle pencil marks."

/**
 * Hint for grid navigation.
 */
const val GRID_NAVIGATION_HINT = "Swipe to navigate cells. " +
        "Double tap to select. Use keyboard arrows for navigation."
