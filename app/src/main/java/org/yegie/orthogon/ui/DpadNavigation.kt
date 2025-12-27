/*
 * DpadNavigation.kt: D-pad navigation utilities for Android TV
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 *
 * Provides D-pad (directional pad) navigation support for the game grid,
 * enabling play on Android TV with a remote control.
 */

package org.yegie.orthogon.ui

import android.view.KeyEvent
import androidx.compose.foundation.focusable
import androidx.compose.foundation.layout.Box
import androidx.compose.runtime.*
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.input.key.*
import androidx.compose.ui.platform.LocalContext
import android.content.res.Configuration

/**
 * D-pad key event result.
 */
sealed interface DpadAction {
    /** Move selection in a direction */
    data class Move(val dx: Int, val dy: Int) : DpadAction

    /** Input a digit (1-9) */
    data class InputDigit(val digit: Int) : DpadAction

    /** Clear/delete current cell */
    data object Clear : DpadAction

    /** Toggle notes mode */
    data object ToggleNotes : DpadAction

    /** Confirm/select current cell */
    data object Select : DpadAction

    /** Request hint */
    data object Hint : DpadAction

    /** Open menu */
    data object Menu : DpadAction

    /** Undo last action */
    data object Undo : DpadAction

    /** Unhandled key */
    data object Unhandled : DpadAction
}

/**
 * Maps key events to D-pad actions.
 * Supports:
 * - Arrow keys for navigation
 * - Number keys (1-9) for digit input
 * - Enter/Select for confirmation
 * - Backspace/Delete for clearing
 * - Menu key for options
 */
fun KeyEvent.toDpadAction(): DpadAction {
    return when (keyCode) {
        // Navigation
        KeyEvent.KEYCODE_DPAD_UP -> DpadAction.Move(0, -1)
        KeyEvent.KEYCODE_DPAD_DOWN -> DpadAction.Move(0, 1)
        KeyEvent.KEYCODE_DPAD_LEFT -> DpadAction.Move(-1, 0)
        KeyEvent.KEYCODE_DPAD_RIGHT -> DpadAction.Move(1, 0)

        // Digit input (numpad and main keyboard)
        KeyEvent.KEYCODE_1, KeyEvent.KEYCODE_NUMPAD_1 -> DpadAction.InputDigit(1)
        KeyEvent.KEYCODE_2, KeyEvent.KEYCODE_NUMPAD_2 -> DpadAction.InputDigit(2)
        KeyEvent.KEYCODE_3, KeyEvent.KEYCODE_NUMPAD_3 -> DpadAction.InputDigit(3)
        KeyEvent.KEYCODE_4, KeyEvent.KEYCODE_NUMPAD_4 -> DpadAction.InputDigit(4)
        KeyEvent.KEYCODE_5, KeyEvent.KEYCODE_NUMPAD_5 -> DpadAction.InputDigit(5)
        KeyEvent.KEYCODE_6, KeyEvent.KEYCODE_NUMPAD_6 -> DpadAction.InputDigit(6)
        KeyEvent.KEYCODE_7, KeyEvent.KEYCODE_NUMPAD_7 -> DpadAction.InputDigit(7)
        KeyEvent.KEYCODE_8, KeyEvent.KEYCODE_NUMPAD_8 -> DpadAction.InputDigit(8)
        KeyEvent.KEYCODE_9, KeyEvent.KEYCODE_NUMPAD_9 -> DpadAction.InputDigit(9)

        // Clear/delete
        KeyEvent.KEYCODE_DEL,
        KeyEvent.KEYCODE_FORWARD_DEL,
        KeyEvent.KEYCODE_0,
        KeyEvent.KEYCODE_NUMPAD_0 -> DpadAction.Clear

        // Select/confirm
        KeyEvent.KEYCODE_DPAD_CENTER,
        KeyEvent.KEYCODE_ENTER,
        KeyEvent.KEYCODE_NUMPAD_ENTER -> DpadAction.Select

        // Notes toggle (N key or color button)
        KeyEvent.KEYCODE_N,
        KeyEvent.KEYCODE_PROG_YELLOW -> DpadAction.ToggleNotes

        // Hint (H key or color button)
        KeyEvent.KEYCODE_H,
        KeyEvent.KEYCODE_PROG_GREEN -> DpadAction.Hint

        // Undo (U key or color button)
        KeyEvent.KEYCODE_U,
        KeyEvent.KEYCODE_PROG_RED -> DpadAction.Undo

        // Menu
        KeyEvent.KEYCODE_MENU,
        KeyEvent.KEYCODE_PROG_BLUE -> DpadAction.Menu

        else -> DpadAction.Unhandled
    }
}

/**
 * Modifier extension for handling D-pad key events.
 */
@OptIn(ExperimentalComposeUiApi::class)
fun Modifier.handleDpadInput(
    onAction: (DpadAction) -> Boolean
): Modifier = this.onKeyEvent { event ->
    if (event.type == KeyEventType.KeyDown) {
        val action = event.nativeKeyEvent.toDpadAction()
        if (action != DpadAction.Unhandled) {
            onAction(action)
        } else {
            false
        }
    } else {
        false
    }
}

/**
 * Grid navigation state for D-pad control.
 */
class GridNavigationState(
    val gridSize: Int,
    initialX: Int = 0,
    initialY: Int = 0
) {
    var selectedX by mutableIntStateOf(initialX.coerceIn(0, gridSize - 1))
        private set
    var selectedY by mutableIntStateOf(initialY.coerceIn(0, gridSize - 1))
        private set

    /**
     * Move selection by delta, wrapping at edges.
     */
    fun move(dx: Int, dy: Int) {
        selectedX = (selectedX + dx + gridSize) % gridSize
        selectedY = (selectedY + dy + gridSize) % gridSize
    }

    /**
     * Set selection to specific cell.
     */
    fun select(x: Int, y: Int) {
        selectedX = x.coerceIn(0, gridSize - 1)
        selectedY = y.coerceIn(0, gridSize - 1)
    }

    /**
     * Check if cell is selected.
     */
    fun isSelected(x: Int, y: Int): Boolean = x == selectedX && y == selectedY
}

/**
 * Remember a grid navigation state.
 */
@Composable
fun rememberGridNavigationState(
    gridSize: Int,
    initialX: Int = 0,
    initialY: Int = 0
): GridNavigationState {
    return remember(gridSize) {
        GridNavigationState(gridSize, initialX, initialY)
    }
}

/**
 * Detect if running on Android TV.
 */
@Composable
fun isAndroidTv(): Boolean {
    val context = LocalContext.current
    return remember(context) {
        @Suppress("DEPRECATION")
        val uiModeManager = context.getSystemService(android.content.Context.UI_MODE_SERVICE) as? android.app.UiModeManager
        uiModeManager?.currentModeType == Configuration.UI_MODE_TYPE_TELEVISION
    }
}

/**
 * Focus state wrapper for TV navigation.
 * Provides visual focus indication and keyboard handling.
 */
@Composable
fun TvFocusableBox(
    modifier: Modifier = Modifier,
    focusRequester: FocusRequester = remember { FocusRequester() },
    onFocused: () -> Unit = {},
    onKeyEvent: (DpadAction) -> Boolean = { false },
    content: @Composable (isFocused: Boolean) -> Unit
) {
    var isFocused by remember { mutableStateOf(false) }

    Box(
        modifier = modifier
            .focusRequester(focusRequester)
            .onFocusChanged { state ->
                isFocused = state.isFocused
                if (state.isFocused) onFocused()
            }
            .handleDpadInput(onKeyEvent)
            .focusable()
    ) {
        content(isFocused)
    }
}

/**
 * TV-specific visual constants.
 */
object TvConstants {
    /** Focus ring width for TV */
    const val FOCUS_RING_WIDTH_DP = 4f

    /** Animation duration for focus transitions */
    const val FOCUS_ANIMATION_MS = 150

    /** Scale factor for focused elements */
    const val FOCUS_SCALE = 1.05f

    /** Minimum touch target for TV (48dp recommended) */
    const val MIN_TOUCH_TARGET_DP = 48
}
