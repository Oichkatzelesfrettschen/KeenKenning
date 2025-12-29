/*
 * GameScreen.kt: Compose UI for the KenKen game board
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.Settings
import org.yegie.keenkenning.data.SaveSlotInfo
import androidx.compose.material3.*
import androidx.compose.material3.HorizontalDivider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.focus.focusTarget
import androidx.compose.ui.input.key.*
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.yegie.keenkenning.ui.theme.*
import org.yegie.keenkenning.data.GameMode
import kotlin.math.ceil
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Convert cell value to display string.
 * For standard grids (1-9), shows decimal digits.
 * For extended grids (10-16), shows hex digits (A-G for 10-16).
 */
private fun valueToDisplay(value: Int): String {
    return when {
        value <= 9 -> value.toString()
        value == 10 -> "A"
        value == 11 -> "B"
        value == 12 -> "C"
        value == 13 -> "D"
        value == 14 -> "E"
        value == 15 -> "F"
        value == 16 -> "G"
        else -> value.toString()
    }
}

/**
 * Accessible Game Screen
 *
 * Accessibility features:
 * - WCAG AA compliant contrast ratios (4.5:1 minimum)
 * - Colorblind-safe zone coloring with distinct luminance
 * - Never relies on color alone (thick borders for cages)
 * - 48dp+ touch targets for all interactive elements
 * - Haptic feedback for actions
 * - Screen reader support via semantics
 * - Responsive scaling for all screen sizes
 * - Reduced motion option
 * - Clear visual hierarchy
 */

/**
 * Transforms a color for colorblind accessibility modes.
 * Uses scientifically-validated color transformations for each type of color vision deficiency.
 */
fun Color.forColorblindMode(mode: ColorblindMode): Color {
    return when (mode) {
        ColorblindMode.NORMAL -> this
        ColorblindMode.DEUTERANOPIA -> {
            // Green-blind: shift greens toward blue, increase blue/yellow contrast
            val r = red * 0.625f + green * 0.375f
            val g = red * 0.7f + green * 0.3f
            val b = blue
            Color(r.coerceIn(0f, 1f), g.coerceIn(0f, 1f), b.coerceIn(0f, 1f), alpha)
        }
        ColorblindMode.PROTANOPIA -> {
            // Red-blind: shift reds toward orange/yellow
            val r = red * 0.567f + green * 0.433f
            val g = red * 0.558f + green * 0.442f
            val b = blue
            Color(r.coerceIn(0f, 1f), g.coerceIn(0f, 1f), b.coerceIn(0f, 1f), alpha)
        }
        ColorblindMode.TRITANOPIA -> {
            // Blue-blind: increase red/blue distinction, shift blues toward cyan
            val r = red * 0.95f + green * 0.05f
            val g = green * 0.433f + blue * 0.567f
            val b = green * 0.475f + blue * 0.525f
            Color(r.coerceIn(0f, 1f), g.coerceIn(0f, 1f), b.coerceIn(0f, 1f), alpha)
        }
        ColorblindMode.HIGH_CONTRAST -> {
            // Maximum visibility: boost saturation and use stark black/white backgrounds
            val luminance = 0.299f * red + 0.587f * green + 0.114f * blue
            if (luminance > 0.5f) {
                Color.White
            } else {
                // Boost saturation for colored elements
                val avg = (red + green + blue) / 3f
                val boosted = { c: Float -> ((c - avg) * 1.5f + avg).coerceIn(0f, 1f) }
                Color(boosted(red), boosted(green), boosted(blue), alpha)
            }
        }
    }
}

@Composable
fun GameScreen(
    viewModel: GameViewModel,
    onMenuClick: (() -> Unit)? = null  // Optional menu callback for navigation
) {
    val uiState by viewModel.uiState.collectAsState()
    val haptic = LocalHapticFeedback.current
    val configuration = LocalConfiguration.current
    val density = LocalDensity.current

    // Focus requester for keyboard input
    val focusRequester = remember { FocusRequester() }

    // Determine if we're on a large screen (tablet)
    val isLargeScreen = configuration.screenWidthDp > 600

    // Detect Android TV for enhanced D-pad navigation visuals
    val isTv = isAndroidTv()

    // Calculate responsive dimensions
    val screenWidth = configuration.screenWidthDp.dp
    val screenHeight = configuration.screenHeightDp.dp

    // Convert to pixels for victory animation
    val screenWidthPx = with(density) { screenWidth.toPx() }
    val screenHeightPx = with(density) { screenHeight.toPx() }

    // Request focus on launch for keyboard input
    LaunchedEffect(Unit) {
        focusRequester.requestFocus()
    }

    // Determine if 8-bit retro mode is active
    val isRetroMode = uiState.gameMode == GameMode.RETRO_8BIT

    // Wrap with appropriate theme based on mode
    val themeContent: @Composable (@Composable () -> Unit) -> Unit = { content ->
        if (isRetroMode) {
            RetroGameTheme(enabled = true) { content() }
        } else {
            GameTheme(darkTheme = false, largeScreen = isLargeScreen) { content() }
        }
    }

    themeContent {
        val dimensions = LocalGameDimensions.current

        // Background gradient (8-bit uses solid dark background)
        val backgroundModifier = if (isRetroMode) {
            Modifier.background(RetroPalette.black)
        } else {
            Modifier.background(
                Brush.verticalGradient(
                    colors = listOf(Color(0xFF1a1a2e), Color(0xFF16213e))
                )
            )
        }

        Box(
            modifier = Modifier
                .fillMaxSize()
                .then(backgroundModifier)
                .focusRequester(focusRequester)
                .focusTarget()
                .onKeyEvent { event ->
                    if (event.type == KeyEventType.KeyDown) {
                        handleKeyEvent(event, viewModel, uiState.size, haptic)
                    } else false
                }
        ) {
            // Accessibility: Announce victory to screen readers
            if (uiState.isSolved && !uiState.victoryAnimationComplete) {
                AnnounceForAccessibility(
                    message = buildVictoryAnnouncement(
                        gridSize = uiState.size,
                        difficulty = uiState.difficultyName,
                        elapsedSeconds = uiState.elapsedTimeSeconds
                    ),
                    key = "victory_${uiState.size}"
                )
            }

            // Show victory animation if solved and not yet shown
            if (uiState.showVictoryAnimation && !uiState.victoryAnimationComplete && uiState.size > 0) {
                val gridSize = calculateGridSize(
                    screenWidth = screenWidth,
                    screenHeight = screenHeight,
                    puzzleSize = uiState.size,
                    isLargeScreen = isLargeScreen
                )
                val cellSizePx = with(density) { (gridSize / uiState.size).toPx() }
                val gridOffsetX = (screenWidthPx - with(density) { gridSize.toPx() }) / 2
                val gridOffsetY = with(density) { 80.dp.toPx() } // Below top bar

                VictoryAnimation(
                    cells = uiState.cells,
                    gridSize = uiState.size,
                    cellSizePx = cellSizePx,
                    gridOffsetX = gridOffsetX,
                    gridOffsetY = gridOffsetY,
                    screenWidth = screenWidthPx,
                    screenHeight = screenHeightPx,
                    onAnimationComplete = { viewModel.onVictoryAnimationComplete() }
                )
            } else {
                // Normal game UI
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .statusBarsPadding()  // Avoid overlapping Android status bar
                        .padding(dimensions.gridPadding),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    // Top Bar - auto-hides in immersive/focus mode
                    AnimatedVisibility(
                        visible = uiState.uiVisible,
                        enter = slideInVertically(initialOffsetY = { -it }) + fadeIn(),
                        exit = slideOutVertically(targetOffsetY = { -it }) + fadeOut()
                    ) {
                        TopBar(
                            showSmartHints = uiState.showSmartHints,
                            onToggleHints = { viewModel.toggleSmartHints() },
                            onShowInfo = { viewModel.toggleInfoDialog() },
                            onShowSettings = { viewModel.toggleSettingsDialog() },
                            onMenuClick = onMenuClick,
                            gridSize = uiState.size,
                            difficultyName = uiState.difficultyName,
                            elapsedSeconds = uiState.elapsedTimeSeconds,
                            isMlGenerated = uiState.isMlGenerated,
                            showTimer = uiState.showTimer
                        )
                    }

                    Spacer(modifier = Modifier.height(8.dp))

                    // Simple solved indicator (animation handles the epic version)
                    if (uiState.isSolved && uiState.victoryAnimationComplete) {
                        CompactVictoryBanner()
                        Spacer(modifier = Modifier.height(8.dp))
                    }

                    // Game Grid - responsive sizing with pinch-to-zoom for large grids
                    if (uiState.size > 0) {
                        val gridSize = calculateGridSize(
                            screenWidth = screenWidth,
                            screenHeight = screenHeight,
                            puzzleSize = uiState.size,
                            isLargeScreen = isLargeScreen
                        )

                        // Enable pinch-to-zoom only for grids 7x7 and larger
                        val enableZoom = uiState.size >= 7

                        if (enableZoom) {
                            ZoomableGameGrid(
                                state = uiState,
                                gridSize = gridSize,
                                isTv = isTv,
                                onCellClick = { x, y ->
                                    viewModel.onCellClicked(x, y)
                                    haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                                }
                            )
                        } else {
                            AccessibleGameGrid(
                                state = uiState,
                                gridSize = gridSize,
                                isTv = isTv,
                                onCellClick = { x, y ->
                                    viewModel.onCellClicked(x, y)
                                    haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                                }
                            )
                        }
                    } else {
                        Box(
                            modifier = Modifier.weight(1f),
                            contentAlignment = Alignment.Center
                        ) {
                            CircularProgressIndicator(color = Color(0xFF7B68EE))
                        }
                    }

                    Spacer(modifier = Modifier.weight(1f))

                    // Input Controls - auto-hides in immersive/focus mode
                    AnimatedVisibility(
                        visible = uiState.uiVisible,
                        enter = slideInVertically(initialOffsetY = { it }) + fadeIn(),
                        exit = slideOutVertically(targetOffsetY = { it }) + fadeOut()
                    ) {
                        AccessibleInputPad(
                            size = uiState.size,
                            isNoteMode = uiState.isInputtingNotes,
                            isZeroInclusive = uiState.gameMode == org.yegie.keenkenning.data.GameMode.ZERO_INCLUSIVE,
                            isNegativeMode = uiState.gameMode == org.yegie.keenkenning.data.GameMode.NEGATIVE_NUMBERS,
                            showHintButton = uiState.gameMode == org.yegie.keenkenning.data.GameMode.HINT_MODE,
                            onNumberClick = {
                                viewModel.onInput(it)
                                viewModel.onUserInteraction()
                                haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                            },
                            onUndoClick = {
                                viewModel.onUndo()
                                viewModel.onUserInteraction()
                                haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                            },
                            onNoteToggle = {
                                viewModel.toggleNoteMode()
                                viewModel.onUserInteraction()
                                haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                            },
                            onClearClick = {
                                viewModel.clearCell()
                                viewModel.onUserInteraction()
                                haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                            },
                            onHintClick = {
                                viewModel.requestHint()
                                viewModel.onUserInteraction()
                                haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                            }
                        )
                    }

                    Spacer(modifier = Modifier.height(16.dp))
                }

                // Info Dialog
                if (uiState.showInfoDialog) {
                    InfoDialog(onDismiss = { viewModel.toggleInfoDialog() })
                }

                // Settings Dialog
                if (uiState.showSettingsDialog) {
                    SettingsDialog(
                        darkTheme = uiState.darkTheme,
                        fontScale = uiState.fontScale,
                        showSmartHints = uiState.showSmartHints,
                        showTimer = uiState.showTimer,
                        immersiveMode = uiState.immersiveMode,
                        colorblindMode = uiState.colorblindMode,
                        onToggleDarkTheme = { viewModel.toggleDarkTheme() },
                        onFontScaleChange = { viewModel.setFontScale(it) },
                        onToggleSmartHints = { viewModel.toggleSmartHints() },
                        onToggleShowTimer = { viewModel.toggleShowTimer() },
                        onToggleImmersiveMode = { viewModel.toggleImmersiveMode() },
                        onColorblindModeChange = { viewModel.setColorblindMode(it) },
                        onSaveGame = {
                            viewModel.toggleSettingsDialog()
                            viewModel.toggleSaveDialog()
                        },
                        onLoadGame = {
                            viewModel.toggleSettingsDialog()
                            viewModel.toggleLoadDialog()
                        },
                        onDismiss = { viewModel.toggleSettingsDialog() }
                    )
                }

                // Save Dialog
                if (uiState.showSaveDialog) {
                    SaveLoadDialog(
                        isSaveMode = true,
                        onFetchSlots = { viewModel.getSaveSlots() },
                        onSlotSelected = { slotIndex -> viewModel.saveToSlot(slotIndex) },
                        onSlotDeleted = { slotIndex -> viewModel.deleteSlot(slotIndex) },
                        onDismiss = { viewModel.toggleSaveDialog() }
                    )
                }

                // Load Dialog
                if (uiState.showLoadDialog) {
                    SaveLoadDialog(
                        isSaveMode = false,
                        onFetchSlots = { viewModel.getSaveSlots() },
                        onSlotSelected = { slotIndex -> viewModel.loadFromSlot(slotIndex) },
                        onSlotDeleted = { slotIndex -> viewModel.deleteSlot(slotIndex) },
                        onDismiss = { viewModel.toggleLoadDialog() }
                    )
                }

                // Hint Dialog (Phase 4a: Tutorial Mode)
                val currentHint = uiState.currentHint
                if (uiState.showHintDialog && currentHint != null) {
                    HintDialog(
                        hint = currentHint,
                        onDismiss = { viewModel.dismissHint() },
                        onApply = { viewModel.applyHint() }
                    )
                }

                // Tap-to-reveal overlay when UI is hidden in immersive mode
                if (uiState.immersiveMode && !uiState.uiVisible) {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .clickable(
                                indication = null,
                                interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() }
                            ) {
                                viewModel.showUi()
                            }
                    )
                }
            }
        }
    }
}

/**
 * Handle keyboard and game controller events for accessibility.
 *
 * Keyboard support:
 * - Arrow keys/WASD: Navigation
 * - 1-9: Number input
 * - N/Space: Toggle notes mode
 * - U/Z: Undo
 * - Delete/Backspace/0: Clear cell
 *
 * Game controller support (D-pad + buttons):
 * - D-pad: Navigation
 * - A button (BUTTON_A): Toggle notes mode
 * - B button (BUTTON_B): Undo
 * - X button (BUTTON_X): Clear cell
 * - Y button (BUTTON_Y): Request hint
 * - Shoulder L/R: Cycle through numbers (for grids > 9)
 * - Start: Toggle settings
 */
private fun handleKeyEvent(
    event: KeyEvent,
    viewModel: GameViewModel,
    size: Int,
    haptic: androidx.compose.ui.hapticfeedback.HapticFeedback
): Boolean {
    return when (event.key) {
        // Arrow key / D-pad navigation
        Key.DirectionUp, Key.W -> {
            viewModel.moveSelection(0, -1)
            haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
            true
        }
        Key.DirectionDown, Key.S -> {
            viewModel.moveSelection(0, 1)
            haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
            true
        }
        Key.DirectionLeft, Key.A -> {
            viewModel.moveSelection(-1, 0)
            haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
            true
        }
        Key.DirectionRight, Key.D -> {
            viewModel.moveSelection(1, 0)
            haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
            true
        }

        // Number input (1-9)
        Key.One, Key.NumPad1 -> { if (size >= 1) { viewModel.onInput(1); true } else false }
        Key.Two, Key.NumPad2 -> { if (size >= 2) { viewModel.onInput(2); true } else false }
        Key.Three, Key.NumPad3 -> { if (size >= 3) { viewModel.onInput(3); true } else false }
        Key.Four, Key.NumPad4 -> { if (size >= 4) { viewModel.onInput(4); true } else false }
        Key.Five, Key.NumPad5 -> { if (size >= 5) { viewModel.onInput(5); true } else false }
        Key.Six, Key.NumPad6 -> { if (size >= 6) { viewModel.onInput(6); true } else false }
        Key.Seven, Key.NumPad7 -> { if (size >= 7) { viewModel.onInput(7); true } else false }
        Key.Eight, Key.NumPad8 -> { if (size >= 8) { viewModel.onInput(8); true } else false }
        Key.Nine, Key.NumPad9 -> { if (size >= 9) { viewModel.onInput(9); true } else false }

        // Toggle notes mode (N, Space, or controller A button)
        Key.N, Key.Spacebar, Key.ButtonA -> {
            viewModel.toggleNoteMode()
            haptic.performHapticFeedback(HapticFeedbackType.LongPress)
            true
        }

        // Undo (U, Z, or controller B button)
        Key.U, Key.Z, Key.ButtonB -> {
            viewModel.onUndo()
            haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
            true
        }

        // Clear cell (Delete, Backspace, 0, or controller X button)
        Key.Delete, Key.Backspace, Key.Zero, Key.ButtonX -> {
            viewModel.clearCell()
            haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
            true
        }

        // Request hint (controller Y button)
        Key.ButtonY, Key.H -> {
            viewModel.requestHint()
            haptic.performHapticFeedback(HapticFeedbackType.LongPress)
            true
        }

        // Toggle settings (controller Start button or Escape)
        Key.ButtonStart, Key.Escape -> {
            viewModel.toggleSettingsDialog()
            true
        }

        else -> false
    }
}

/**
 * Minimal single-row TopBar for immersive gameplay.
 * Design: [Back] | [Badges] | [Timer] | [Menu]
 *
 * Accessibility: All interactive elements have 48dp+ touch targets and semantic descriptions.
 * Cognitive load: Reduced from 2 rows to 1, removed redundant title.
 */
@Composable
private fun TopBar(
    showSmartHints: Boolean,
    onToggleHints: () -> Unit,
    onShowInfo: () -> Unit,
    onShowSettings: () -> Unit,
    onMenuClick: (() -> Unit)? = null,
    gridSize: Int = 0,
    difficultyName: String = "Easy",
    elapsedSeconds: Long = 0,
    isMlGenerated: Boolean = false,
    showTimer: Boolean = true  // New: allow hiding timer to reduce anxiety
) {
    // Dropdown menu state
    var showMenu by remember { mutableStateOf(false) }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Left: Back button
        if (onMenuClick != null) {
            IconButton(
                onClick = onMenuClick,
                modifier = Modifier
                    .size(48.dp)
                    .semantics { contentDescription = "Return to main menu" }
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = null,
                    tint = Color(0xFF7B68EE),
                    modifier = Modifier.size(24.dp)
                )
            }
        } else {
            Spacer(modifier = Modifier.size(48.dp))
        }

        // Center-left: Compact badges
        Row(
            horizontalArrangement = Arrangement.spacedBy(4.dp),
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.weight(1f)
        ) {
            // Combined badge: "6x6 Insane" or "6x6 Insane AI"
            Surface(
                shape = RoundedCornerShape(6.dp),
                color = getDifficultyColor(difficultyName).copy(alpha = 0.9f)
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    if (gridSize > 0) {
                        Text(
                            text = "${gridSize}x${gridSize}",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        )
                    }
                    Text(
                        text = difficultyName,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Medium,
                        color = Color.White.copy(alpha = 0.9f)
                    )
                    if (isMlGenerated) {
                        Text(
                            text = "ML",
                            fontSize = 10.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFF00CED1)
                        )
                    }
                }
            }
        }

        // Center-right: Timer (optional)
        if (showTimer) {
            val timerDescription = "Elapsed time: ${formatTimeForSpeech(elapsedSeconds)}"
            Surface(
                shape = RoundedCornerShape(6.dp),
                color = Color(0xFF2a2a4a),
                modifier = Modifier.semantics {
                    contentDescription = timerDescription
                    if (elapsedSeconds % 60 == 0L && elapsedSeconds > 0) {
                        liveRegion = LiveRegionMode.Polite
                    }
                }
            ) {
                Text(
                    text = formatTime(elapsedSeconds),
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Medium,
                    color = Color(0xFFE0E0E0),
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                    fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace
                )
            }
        }

        // Right: Overflow menu
        Box {
            IconButton(
                onClick = { showMenu = true },
                modifier = Modifier
                    .size(48.dp)
                    .semantics { contentDescription = "Open game menu" }
            ) {
                Icon(
                    imageVector = Icons.Default.Menu,
                    contentDescription = null,
                    tint = Color(0xFF7B68EE),
                    modifier = Modifier.size(24.dp)
                )
            }

            DropdownMenu(
                expanded = showMenu,
                onDismissRequest = { showMenu = false }
            ) {
                DropdownMenuItem(
                    text = {
                        Text(if (showSmartHints) "Disable ML Hints" else "Enable ML Hints")
                    },
                    onClick = {
                        onToggleHints()
                        showMenu = false
                    },
                    leadingIcon = {
                        Icon(
                            imageVector = if (showSmartHints) Icons.Default.Check else Icons.Default.Refresh,
                            contentDescription = null,
                            tint = if (showSmartHints) Color(0xFF00CED1) else Color.Gray
                        )
                    }
                )
                DropdownMenuItem(
                    text = { Text("Settings") },
                    onClick = {
                        onShowSettings()
                        showMenu = false
                    },
                    leadingIcon = {
                        Icon(Icons.Default.Settings, contentDescription = null)
                    }
                )
                DropdownMenuItem(
                    text = { Text("Game Info") },
                    onClick = {
                        onShowInfo()
                        showMenu = false
                    },
                    leadingIcon = {
                        Icon(Icons.Default.Info, contentDescription = null)
                    }
                )
            }
        }
    }
}

private fun formatTime(seconds: Long): String {
    val mins = seconds / 60
    val secs = seconds % 60
    return "%02d:%02d".format(mins, secs)
}

private fun getDifficultyColor(difficulty: String): Color = when (difficulty) {
    "Easy" -> Color(0xFF4CAF50)      // Green
    "Normal" -> Color(0xFFFFC107)    // Amber
    "Hard" -> Color(0xFFFF9800)      // Orange
    "Insane" -> Color(0xFFF44336)    // Red - hard but logically solvable
    "Ludicrous" -> Color(0xFF9C27B0) // Purple - may require guessing
    else -> Color(0xFF666666)
}

@Composable
private fun CompactVictoryBanner() {
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .semantics {
                liveRegion = LiveRegionMode.Assertive
                contentDescription = "Puzzle solved! Congratulations!"
            },
        color = Color(0xFF2E7D32).copy(alpha = 0.9f),
        shape = RoundedCornerShape(8.dp)
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = Icons.Default.Check,
                contentDescription = null,
                tint = Color.White,
                modifier = Modifier.size(24.dp)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = "PUZZLE SOLVED!",
                color = Color.White,
                fontWeight = FontWeight.Bold,
                fontSize = 18.sp,
                letterSpacing = 2.sp
            )
        }
    }
}

@Suppress("UNUSED_PARAMETER")  // isLargeScreen reserved for responsive layout
@Composable
private fun calculateGridSize(
    screenWidth: Dp,
    screenHeight: Dp,
    puzzleSize: Int,
    isLargeScreen: Boolean
): Dp {
    // Reserve space for header (~50dp), controls (~180dp), and padding
    val availableHeight = screenHeight - 250.dp
    val availableWidth = screenWidth - 16.dp

    // Use the smaller dimension to ensure square grid fits
    val maxGridSize = min(availableWidth.value, availableHeight.value).dp

    // Ensure minimum cell size for touch targets (48dp)
    val minGridSize = (48 * puzzleSize).dp

    return maxGridSize.coerceAtLeast(minGridSize)
}

@Composable
private fun AccessibleGameGrid(
    state: GameUiState,
    gridSize: Dp,
    isTv: Boolean = false,
    onCellClick: (Int, Int) -> Unit
) {
    val size = state.size

    // Outer border for entire grid
    Surface(
        modifier = Modifier
            .size(gridSize)
            .semantics { contentDescription = "${size}x${size} Keen puzzle grid" },
        color = Color.White,
        shape = RoundedCornerShape(4.dp),
        shadowElevation = 4.dp
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .border(3.dp, Color(0xFF212121), RoundedCornerShape(4.dp))
                .padding(1.dp)
        ) {
            Column(modifier = Modifier.fillMaxSize()) {
                for (y in 0 until size) {
                    Row(modifier = Modifier.weight(1f)) {
                        for (x in 0 until size) {
                            val cell = state.cells[x][y]
                            AccessibleCellView(
                                cell = cell,
                                showHints = state.showSmartHints,
                                puzzleSize = size,
                                colorblindMode = state.colorblindMode,
                                isTv = isTv,
                                modifier = Modifier
                                    .weight(1f)
                                    .fillMaxHeight(),
                                onClick = { onCellClick(x, y) }
                            )
                        }
                    }
                }
            }
        }
    }
}

/**
 * Zoomable wrapper for game grid - enables pinch-to-zoom for large puzzles (7x7+)
 * Provides 1x-2.5x zoom range with pan support when zoomed in
 */
@Composable
private fun ZoomableGameGrid(
    state: GameUiState,
    gridSize: Dp,
    isTv: Boolean = false,
    onCellClick: (Int, Int) -> Unit
) {
    var scale by remember { mutableFloatStateOf(1f) }
    var offsetX by remember { mutableFloatStateOf(0f) }
    var offsetY by remember { mutableFloatStateOf(0f) }

    val minScale = 1f
    val maxScale = 2.5f

    Box(
        modifier = Modifier
            .size(gridSize)
            .clip(RoundedCornerShape(4.dp))
            .pointerInput(Unit) {
                detectTransformGestures { _, pan, zoom, _ ->
                    // Apply zoom
                    val newScale = (scale * zoom).coerceIn(minScale, maxScale)
                    scale = newScale

                    // Apply pan only when zoomed in
                    if (scale > 1f) {
                        val maxOffset = (gridSize.toPx() * (scale - 1f)) / 2f
                        offsetX = (offsetX + pan.x).coerceIn(-maxOffset, maxOffset)
                        offsetY = (offsetY + pan.y).coerceIn(-maxOffset, maxOffset)
                    } else {
                        offsetX = 0f
                        offsetY = 0f
                    }
                }
            },
        contentAlignment = Alignment.Center
    ) {
        Box(
            modifier = Modifier
                .graphicsLayer(
                    scaleX = scale,
                    scaleY = scale,
                    translationX = offsetX,
                    translationY = offsetY
                )
        ) {
            AccessibleGameGridContent(
                state = state,
                gridSize = gridSize,
                isTv = isTv,
                onCellClick = onCellClick
            )
        }

        // Zoom indicator when zoomed
        if (scale > 1.05f) {
            Surface(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(4.dp),
                color = Color(0x88000000),
                shape = RoundedCornerShape(4.dp)
            ) {
                Text(
                    text = "${(scale * 100).toInt()}%",
                    fontSize = 10.sp,
                    color = Color.White,
                    modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                )
            }
        }
    }
}

/**
 * Inner grid content - shared between zoomable and non-zoomable versions
 */
@Composable
private fun AccessibleGameGridContent(
    state: GameUiState,
    gridSize: Dp,
    isTv: Boolean = false,
    onCellClick: (Int, Int) -> Unit
) {
    val size = state.size

    Surface(
        modifier = Modifier
            .size(gridSize)
            .semantics { contentDescription = "${size}x${size} Keen puzzle grid" },
        color = Color.White,
        shape = RoundedCornerShape(4.dp),
        shadowElevation = 4.dp
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .border(3.dp, Color(0xFF212121), RoundedCornerShape(4.dp))
                .padding(1.dp)
        ) {
            Column(modifier = Modifier.fillMaxSize()) {
                for (y in 0 until size) {
                    Row(modifier = Modifier.weight(1f)) {
                        for (x in 0 until size) {
                            val cell = state.cells[x][y]
                            AccessibleCellView(
                                cell = cell,
                                showHints = state.showSmartHints,
                                puzzleSize = size,
                                colorblindMode = state.colorblindMode,
                                isTv = isTv,
                                modifier = Modifier
                                    .weight(1f)
                                    .fillMaxHeight(),
                                onClick = { onCellClick(x, y) }
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun AccessibleCellView(
    cell: UiCell,
    showHints: Boolean,
    puzzleSize: Int,
    colorblindMode: ColorblindMode,
    isTv: Boolean = false,
    modifier: Modifier,
    onClick: () -> Unit
) {
    val dimensions = LocalGameDimensions.current
    val colors = LocalGameColors.current

    // Zone-based background color for visual grouping
    // Uses varied luminance so it works in grayscale too
    // Transformed for colorblind accessibility when mode is active
    val zoneColor = remember(cell.zoneId, colorblindMode) {
        ZoneColors.forZone(cell.zoneId, highContrast = colorblindMode == ColorblindMode.HIGH_CONTRAST)
            .forColorblindMode(colorblindMode)
    }

    // Selection uses both color AND thick border (never color alone)
    // TV mode: extra-thick border for visibility on large screens
    val backgroundColor by animateColorAsState(
        targetValue = if (cell.isSelected) Color(0xFFD1C4E9) else zoneColor,
        animationSpec = tween(150),
        label = "cellBg"
    )

    // TV mode: 4dp focus ring for visibility at 10-foot viewing distance
    val selectionBorderWidth = when {
        cell.isSelected && isTv -> 4.dp  // Thick TV focus ring
        cell.isSelected -> 3.dp          // Standard mobile selection
        else -> 0.dp
    }
    // TV mode: high-contrast cyan for focus ring
    val selectionBorderColor = if (isTv) Color(0xFF00CED1) else Color(0xFF5E35B1)

    // Cage borders - thick black lines provide non-color differentiation
    val cageBorderColor = Color(0xFF212121)
    val gridLineColor = Color(0xFFBDBDBD)
    val cageBorderWidth = dimensions.cageBorderWidth
    val gridBorderWidth = dimensions.gridBorderWidth

    // Calculate text sizes based on puzzle size for readability
    val valueTextSize = when {
        puzzleSize <= 4 -> 28.sp
        puzzleSize <= 6 -> 24.sp
        puzzleSize <= 8 -> 20.sp
        else -> 18.sp
    }

    val clueTextSize = when {
        puzzleSize <= 4 -> 12.sp
        puzzleSize <= 6 -> 11.sp
        puzzleSize <= 8 -> 10.sp
        else -> 9.sp
    }

    val noteTextSize = when {
        puzzleSize <= 4 -> 10.sp
        puzzleSize <= 6 -> 9.sp
        else -> 8.sp
    }

    BoxWithConstraints(
        contentAlignment = Alignment.Center,
        modifier = modifier
            .background(backgroundColor)
            .then(
                if (cell.isSelected) {
                    Modifier.border(selectionBorderWidth, selectionBorderColor)
                } else Modifier
            )
            .clickable(
                onClickLabel = "Select cell at row ${cell.y + 1}, column ${cell.x + 1}"
            ) { onClick() }
            .drawBehind {
                val w = size.width
                val h = size.height

                // Draw cage borders (thick) or grid lines (thin)
                // Top
                drawLine(
                    color = if (cell.borders.top) cageBorderColor else gridLineColor,
                    start = Offset(0f, 0f),
                    end = Offset(w, 0f),
                    strokeWidth = if (cell.borders.top) cageBorderWidth.toPx() else gridBorderWidth.toPx(),
                    cap = StrokeCap.Square
                )

                // Bottom
                drawLine(
                    color = if (cell.borders.bottom) cageBorderColor else gridLineColor,
                    start = Offset(0f, h),
                    end = Offset(w, h),
                    strokeWidth = if (cell.borders.bottom) cageBorderWidth.toPx() else gridBorderWidth.toPx(),
                    cap = StrokeCap.Square
                )

                // Left
                drawLine(
                    color = if (cell.borders.left) cageBorderColor else gridLineColor,
                    start = Offset(0f, 0f),
                    end = Offset(0f, h),
                    strokeWidth = if (cell.borders.left) cageBorderWidth.toPx() else gridBorderWidth.toPx(),
                    cap = StrokeCap.Square
                )

                // Right
                drawLine(
                    color = if (cell.borders.right) cageBorderColor else gridLineColor,
                    start = Offset(w, 0f),
                    end = Offset(w, h),
                    strokeWidth = if (cell.borders.right) cageBorderWidth.toPx() else gridBorderWidth.toPx(),
                    cap = StrokeCap.Square
                )
            }
            .padding(dimensions.cellPadding)
            .semantics {
                contentDescription = buildCellDescription(cell, puzzleSize)
            }
    ) {
        // Cell dimensions available for proportional calculations
        val cellHeight = maxHeight
        val cellWidth = maxWidth

        // Calculate proportional offsets based on actual cell size
        val valueOffset = cellHeight * dimensions.valueVerticalOffsetRatio
        val noteGridOffset = cellHeight * dimensions.noteGridOffsetRatio
        val hintGridOffset = cellHeight * dimensions.hintGridOffsetRatio

        // Layout zones to prevent overlap:
        // - TopStart: Clue (cage constraint like "20x")
        // - Center/BottomCenter: Main value or Smart Hints
        // - BottomEnd: User notes (compact)

        val hasClue = cell.clue != null

        // Clue (cage constraint) - elevated surface for depth perception
        // Uses theme colors for dark/light mode compatibility
        if (cell.clue != null) {
            Surface(
                modifier = Modifier.align(Alignment.TopStart),
                shape = RoundedCornerShape(dimensions.clueBoxCornerRadius),
                color = colors.surfaceDim,  // Theme-aware surface color
                shadowElevation = dimensions.clueBoxElevation,
                tonalElevation = 1.dp
            ) {
                Text(
                    text = cell.clue,
                    fontSize = clueTextSize,
                    fontWeight = FontWeight.Bold,
                    color = colors.textPrimary,  // Theme-aware text color
                    modifier = Modifier.padding(
                        horizontal = dimensions.clueBoxPaddingHorizontal,
                        vertical = dimensions.clueBoxPaddingVertical
                    )
                )
            }
        }

        // Cell value - centered with proportional offset if clue present
        if (cell.value != null && cell.value != -1) {
            Text(
                text = valueToDisplay(cell.value),
                fontSize = valueTextSize,
                fontWeight = FontWeight.Bold,
                color = colors.textPrimary,  // Theme-aware text color
                textAlign = TextAlign.Center,
                modifier = Modifier
                    .align(Alignment.Center)
                    .padding(top = if (hasClue) valueOffset else 0.dp)
            )
        } else {
            // Show Smart Hints or user notes - with proportional offsets
            if (showHints && cell.smartHintProbs != null) {
                SmartHintsGrid(
                    probs = cell.smartHintProbs,
                    textSize = noteTextSize,
                    hasClue = hasClue,
                    cellWidth = cellWidth,
                    verticalOffset = hintGridOffset
                )
            } else if (cell.notes.isNotEmpty() && cell.notes.any { it }) {
                AccessibleNoteGrid(
                    notes = cell.notes,
                    textSize = noteTextSize,
                    puzzleSize = puzzleSize,
                    hasClue = hasClue,
                    cellWidth = cellWidth,
                    verticalOffset = noteGridOffset,
                    noteBoxSizeRatio = dimensions.noteBoxSizeRatio
                )
            }
        }
    }
}

private fun buildCellDescription(cell: UiCell, puzzleSize: Int = 9): String {
    return buildCellAccessibilityDescription(
        x = cell.x,
        y = cell.y,
        value = cell.value,
        clue = cell.clue,
        isSelected = cell.isSelected,
        notes = cell.notes,
        puzzleSize = puzzleSize
    )
}

/**
 * Smart Hints Grid - displays ML-computed probability hints
 * Uses both opacity AND font weight to indicate confidence level
 * (provides non-color indication for accessibility)
 * Always centered - only shown when no value is entered
 *
 * @param probs List of probabilities for each digit
 * @param textSize Base text size for hints
 * @param hasClue Whether this cell has a cage clue (affects positioning)
 * @param cellWidth Actual cell width for proportional sizing
 * @param verticalOffset Proportional offset when clue is present
 */
@Composable
private fun BoxWithConstraintsScope.SmartHintsGrid(
    probs: List<Float>,
    textSize: androidx.compose.ui.unit.TextUnit,
    hasClue: Boolean = false,
    cellWidth: Dp = 0.dp,
    verticalOffset: Dp = 0.dp
) {
    val colors = LocalGameColors.current
    val count = probs.size
    val gridDim = ceil(sqrt(count.toFloat())).toInt().coerceAtLeast(1)

    // Smaller text for hints to fit better
    val hintSize = (textSize.value * 0.85f).sp

    // Calculate proportional box size based on cell width
    val hintBoxSize = if (cellWidth > 0.dp) {
        (cellWidth / gridDim) * 0.9f  // 90% to leave small gaps
    } else {
        8.dp  // Fallback
    }

    // Center position - use proportional offset when clue present
    Column(
        modifier = Modifier
            .align(Alignment.Center)
            .padding(top = if (hasClue) verticalOffset else 0.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        for (row in 0 until gridDim) {
            Row(
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                for (col in 0 until gridDim) {
                    val num = row * gridDim + col + 1
                    if (num <= count) {
                        val p = probs[num - 1]
                        // Fixed-size box for consistent grid alignment
                        Box(
                            contentAlignment = Alignment.Center,
                            modifier = Modifier.size(hintBoxSize)
                        ) {
                            if (p > 0.1f) {
                                // Use both opacity AND font weight for probability
                                // This provides non-color indication of confidence
                                Text(
                                    text = num.toString(),
                                    fontSize = hintSize,
                                    color = colors.hintText.copy(alpha = (p * 1.2f).coerceIn(0.3f, 1f)),
                                    fontWeight = if (p > 0.7f) FontWeight.Bold else FontWeight.Normal,
                                    textAlign = TextAlign.Center
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * User notes grid - displays pencil marks in a positional grid
 * For 1-9: 3x3 grid, for 10-16: 4x4 grid
 * Grid positions:
 *   1 2 3
 *   4 5 6
 *   7 8 9
 * Centered in the cell - only shown when no value is entered
 *
 * @param notes Boolean list indicating which notes are set
 * @param textSize Base text size (used for scaling)
 * @param puzzleSize Grid size (determines number of digits)
 * @param hasClue Whether this cell has a cage clue (affects positioning)
 * @param cellWidth Actual cell width for proportional box sizing
 * @param verticalOffset Proportional offset when clue is present
 * @param noteBoxSizeRatio Ratio of cell width for each note box
 */
@Composable
private fun BoxWithConstraintsScope.AccessibleNoteGrid(
    notes: List<Boolean>,
    @Suppress("UNUSED_PARAMETER") textSize: androidx.compose.ui.unit.TextUnit,
    puzzleSize: Int,
    hasClue: Boolean = false,
    cellWidth: Dp = 0.dp,
    verticalOffset: Dp = 0.dp,
    noteBoxSizeRatio: Float = 0.28f
) {
    val colors = LocalGameColors.current

    // Check if any notes are set
    val hasAnyNotes = notes.take(puzzleSize).any { it }
    if (!hasAnyNotes) return

    // Calculate grid dimensions based on puzzle size
    // For sizes 1-9: 3x3 grid, for 10-16: 4x4 grid
    val gridDim = when {
        puzzleSize <= 9 -> 3
        puzzleSize <= 16 -> 4
        else -> 5
    }

    // Calculate proportional box size based on cell width
    val noteBoxSize = if (cellWidth > 0.dp) {
        cellWidth * noteBoxSizeRatio
    } else {
        // Fallback to fixed sizes if cellWidth not available
        when {
            puzzleSize <= 6 -> 10.dp
            puzzleSize <= 9 -> 8.dp
            else -> 7.dp
        }
    }

    // Smaller text for notes - scales with puzzle size
    val noteSize = when {
        puzzleSize <= 4 -> 10.sp
        puzzleSize <= 6 -> 9.sp
        puzzleSize <= 9 -> 7.sp
        else -> 6.sp
    }

    // Display as positional grid - use proportional offset when clue present
    Column(
        modifier = Modifier
            .align(Alignment.Center)
            .padding(top = if (hasClue) verticalOffset else 0.dp)
            .background(colors.surface.copy(alpha = 0.5f), RoundedCornerShape(2.dp))
            .padding(1.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        for (row in 0 until gridDim) {
            Row(
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                for (col in 0 until gridDim) {
                    val num = row * gridDim + col + 1  // 1-indexed
                    val noteIndex = num - 1  // 0-indexed for array

                    // Proportionally-sized box for consistent grid alignment
                    Box(
                        contentAlignment = Alignment.Center,
                        modifier = Modifier.size(noteBoxSize)
                    ) {
                        // Only show digit if it's within puzzle range AND note is set
                        if (num <= puzzleSize && noteIndex < notes.size && notes[noteIndex]) {
                            Text(
                                text = if (num <= 9) num.toString() else ('A' + (num - 10)).toString(),
                                fontSize = noteSize,
                                color = colors.noteText,  // Theme-aware note color
                                fontWeight = FontWeight.Medium,
                                textAlign = TextAlign.Center
                            )
                        }
                        // Empty positions just show nothing (box reserves space)
                    }
                }
            }
        }
    }
}

@Composable
private fun AccessibleInputPad(
    size: Int,
    isNoteMode: Boolean,
    isZeroInclusive: Boolean = false,
    isNegativeMode: Boolean = false,
    showHintButton: Boolean = false,
    onNumberClick: (Int) -> Unit,
    onUndoClick: () -> Unit,
    onNoteToggle: () -> Unit,
    onClearClick: () -> Unit,
    onHintClick: () -> Unit = {}
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(12.dp)  // Increased from 8dp for better accessibility
    ) {
        // Number pad: varies by mode
        // Zero-Inclusive: 0 to N-1
        // Negative: -floor(N/2) to -1, 1 to ceil(N/2) (skipping 0)
        // Standard: 1 to N
        val numbers = when {
            isZeroInclusive -> (0 until size).toList()
            isNegativeMode -> {
                val negCount = size / 2
                val negatives = (-negCount until 0).toList()  // -negCount..-1
                val positives = (1..(size - negCount)).toList()  // 1..ceil(N/2)
                negatives + positives
            }
            else -> (1..size).toList()
        }
        val maxPerRow = if (size <= 5) size else 5
        val chunked = numbers.chunked(maxPerRow)

        chunked.forEach { rowNums ->
            Row(
                horizontalArrangement = Arrangement.spacedBy(12.dp),  // Increased from 8dp
                verticalAlignment = Alignment.CenterVertically
            ) {
                rowNums.forEach { num ->
                    NumberButton(
                        number = num,
                        onClick = { onNumberClick(num) }
                    )
                }
            }
        }

        // Visual separator - creates clear grouping between numbers and tools
        Spacer(modifier = Modifier.height(12.dp))
        Box(
            modifier = Modifier
                .width(200.dp)
                .height(1.dp)
                .background(Color(0xFF4a4a6a))
        )
        Spacer(modifier = Modifier.height(12.dp))

        // Tool buttons row - Undo, Clear, Notes toggle
        Row(
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Undo button with icon + text
            Button(
                onClick = onUndoClick,
                modifier = Modifier
                    .height(48.dp)  // WCAG minimum touch target
                    .semantics { contentDescription = "Undo last action" },
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF3a3a5a),
                    contentColor = Color.White
                ),
                shape = RoundedCornerShape(8.dp),
                contentPadding = PaddingValues(horizontal = 12.dp)
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = null,
                    modifier = Modifier.size(18.dp)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text("Undo", fontSize = 13.sp)
            }

            // Clear button - removes value/notes from selected cell
            Button(
                onClick = onClearClick,
                modifier = Modifier
                    .height(48.dp)  // WCAG minimum touch target
                    .semantics { contentDescription = "Clear selected cell" },
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF5a3a3a),
                    contentColor = Color.White
                ),
                shape = RoundedCornerShape(8.dp),
                contentPadding = PaddingValues(horizontal = 12.dp)
            ) {
                Text("×", fontSize = 18.sp, fontWeight = FontWeight.Bold)
                Spacer(modifier = Modifier.width(4.dp))
                Text("Clear", fontSize = 13.sp)
            }

            // Notes toggle - uses text change AND color change
            // State description announces toggle state to screen readers
            Button(
                onClick = onNoteToggle,
                modifier = Modifier
                    .height(48.dp)  // WCAG minimum touch target
                    .semantics {
                        stateDescription = if (isNoteMode) "on" else "off"
                        contentDescription = "Notes mode, double tap to toggle"
                    },
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isNoteMode) Color(0xFF00CED1) else Color(0xFF3a3a5a),
                    contentColor = if (isNoteMode) Color(0xFF1A1A1A) else Color.White
                ),
                shape = RoundedCornerShape(8.dp),
                contentPadding = PaddingValues(horizontal = 12.dp)
            ) {
                Text(
                    text = if (isNoteMode) "Notes: ON" else "Notes: OFF",
                    fontSize = 13.sp,
                    fontWeight = if (isNoteMode) FontWeight.Bold else FontWeight.Normal
                )
            }
        }

        // Hint button row (only in Tutorial/Hint mode)
        if (showHintButton) {
            Spacer(modifier = Modifier.height(8.dp))
            Button(
                onClick = onHintClick,
                modifier = Modifier
                    .height(48.dp)
                    .semantics { contentDescription = "Get a hint for the selected cell" },
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF8B4513),  // Warm brown for hints
                    contentColor = Color.White
                ),
                shape = RoundedCornerShape(8.dp),
                contentPadding = PaddingValues(horizontal = 16.dp)
            ) {
                Icon(
                    imageVector = Icons.Filled.Info,
                    contentDescription = null,
                    modifier = Modifier.size(18.dp)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text("Hint", fontSize = 14.sp, fontWeight = FontWeight.Medium)
            }
        }
    }
}

@Composable
private fun NumberButton(
    number: Int,
    onClick: () -> Unit
) {
    val dimensions = LocalGameDimensions.current
    // Use hex display for extended grids (10-16)
    val displayText = valueToDisplay(number)
    val description = if (number > 9) "$displayText ($number)" else displayText

    // Rounded rectangles reduce cognitive load vs circles (research-backed)
    // 12dp radius is optimal for neurodivergent-friendly design
    Surface(
        modifier = Modifier
            .size(dimensions.buttonMinSize)
            .clip(RoundedCornerShape(12.dp))
            .clickable(onClickLabel = "Enter $description") { onClick() }
            .semantics { contentDescription = "Number $description" },
        color = Color(0xFF2a2a4a),
        shape = RoundedCornerShape(12.dp),
        border = androidx.compose.foundation.BorderStroke(1.dp, Color(0xFF7B68EE))
    ) {
        Box(contentAlignment = Alignment.Center) {
            Text(
                text = displayText,
                fontSize = dimensions.buttonTextSize,
                fontWeight = FontWeight.Medium,
                color = Color(0xFF7B68EE)
            )
        }
    }
}

@Composable
private fun InfoDialog(onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = Color(0xFF2a2a4a),
        titleContentColor = Color(0xFF7B68EE),
        textContentColor = Color(0xFFE0E0E0),
        title = {
            Text(
                "How to Play",
                fontWeight = FontWeight.Bold,
                fontSize = 20.sp
            )
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                InfoSection(
                    title = "Goal",
                    content = "Fill the grid so each row and column contains each number exactly once."
                )
                InfoSection(
                    title = "Cages",
                    content = "Cells with thick borders form cages. The numbers in a cage must combine to match the clue using the given operation (+, -, ×, ÷)."
                )
                InfoSection(
                    title = "Smart Hints",
                    content = "Toggle the check icon to show ML-computed probability hints. Brighter, bolder numbers indicate higher confidence."
                )
                InfoSection(
                    title = "Notes",
                    content = "Use Notes mode to pencil in possible values for a cell."
                )
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("Got it!", color = Color(0xFF7B68EE))
            }
        }
    )
}

@Composable
private fun InfoSection(title: String, content: String) {
    Column {
        Text(
            text = title,
            fontWeight = FontWeight.Bold,
            fontSize = 14.sp,
            color = Color(0xFF7B68EE)
        )
        Text(
            text = content,
            fontSize = 13.sp,
            color = Color(0xFFBDBDBD),
            lineHeight = 18.sp
        )
    }
}

/**
 * Hint Dialog for Tutorial Mode (Phase 4a).
 * Shows ML-suggested digit with reasoning.
 */
@Composable
private fun HintDialog(
    hint: HintInfo,
    onDismiss: () -> Unit,
    onApply: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = Color(0xFF3a2a1a),  // Warm brown theme
        titleContentColor = Color(0xFFDEB887),
        textContentColor = Color(0xFFE0E0E0),
        title = {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Filled.Info,
                    contentDescription = null,
                    tint = Color(0xFFDEB887),
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    "Hint",
                    fontWeight = FontWeight.Bold,
                    fontSize = 20.sp
                )
            }
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                if (hint.suggestedDigit > 0) {
                    // Show suggested digit with confidence
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text("Suggested:", fontSize = 14.sp, color = Color(0xFFBDBDBD))
                        Text(
                            text = hint.suggestedDigit.toString(),
                            fontSize = 28.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFF00CED1)
                        )
                        // Confidence bar
                        Box(
                            modifier = Modifier
                                .width(60.dp)
                                .height(8.dp)
                                .background(Color(0xFF3a3a3a), RoundedCornerShape(4.dp))
                        ) {
                            Box(
                                modifier = Modifier
                                    .fillMaxHeight()
                                    .fillMaxWidth(hint.confidence)
                                    .background(
                                        when {
                                            hint.confidence >= 0.8f -> Color(0xFF00CED1)
                                            hint.confidence >= 0.5f -> Color(0xFFFFD700)
                                            else -> Color(0xFFFF6B6B)
                                        },
                                        RoundedCornerShape(4.dp)
                                    )
                            )
                        }
                    }
                }
                Text(
                    text = hint.reasoning,
                    fontSize = 14.sp,
                    color = Color(0xFFE0E0E0),
                    lineHeight = 20.sp
                )
            }
        },
        confirmButton = {
            if (hint.suggestedDigit > 0 && hint.cellX >= 0) {
                TextButton(onClick = onApply) {
                    Text("Apply", color = Color(0xFF00CED1), fontWeight = FontWeight.Bold)
                }
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Close", color = Color(0xFFDEB887))
            }
        }
    )
}

@Suppress("UNUSED_PARAMETER")  // fontScale/onFontScaleChange reserved for future font scaling UI
@Composable
private fun SettingsDialog(
    darkTheme: Boolean,
    fontScale: Float,
    showSmartHints: Boolean,
    showTimer: Boolean,
    immersiveMode: Boolean,
    colorblindMode: ColorblindMode,
    onToggleDarkTheme: () -> Unit,
    onFontScaleChange: (Float) -> Unit,
    onToggleSmartHints: () -> Unit,
    onToggleShowTimer: () -> Unit,
    onToggleImmersiveMode: () -> Unit,
    onColorblindModeChange: (ColorblindMode) -> Unit,
    onSaveGame: () -> Unit,
    onLoadGame: () -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = Color(0xFF2a2a4a),
        titleContentColor = Color(0xFF7B68EE),
        textContentColor = Color(0xFFE0E0E0),
        title = {
            Text(
                "Settings",
                fontWeight = FontWeight.Bold,
                fontSize = 20.sp
            )
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                // Save/Load buttons row
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Button(
                        onClick = onSaveGame,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF4CAF50)
                        ),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("Save Game", fontSize = 14.sp, color = Color.White)
                    }
                    Button(
                        onClick = onLoadGame,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF2196F3)
                        ),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("Load Game", fontSize = 14.sp, color = Color.White)
                    }
                }

                HorizontalDivider(color = Color(0xFF4a4a6a), thickness = 1.dp)

                // Smart Hints toggle (primary feature)
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            text = "Smart Hints",
                            fontWeight = FontWeight.Medium,
                            fontSize = 14.sp,
                            color = Color(0xFFE0E0E0)
                        )
                        Text(
                            text = "Show ML-computed probability hints",
                            fontSize = 12.sp,
                            color = Color(0xFF999999)
                        )
                    }
                    Switch(
                        checked = showSmartHints,
                        onCheckedChange = { onToggleSmartHints() },
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = Color(0xFF00CED1),
                            checkedTrackColor = Color(0xFF008B8B),
                            uncheckedThumbColor = Color(0xFF999999),
                            uncheckedTrackColor = Color(0xFF3a3a5a)
                        )
                    )
                }

                // Dark theme toggle
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            text = "Dark Theme",
                            fontWeight = FontWeight.Medium,
                            fontSize = 14.sp,
                            color = Color(0xFFE0E0E0)
                        )
                        Text(
                            text = "Reduce eye strain in low light",
                            fontSize = 12.sp,
                            color = Color(0xFF999999)
                        )
                    }
                    Switch(
                        checked = darkTheme,
                        onCheckedChange = { onToggleDarkTheme() },
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = Color(0xFF7B68EE),
                            checkedTrackColor = Color(0xFF5E35B1),
                            uncheckedThumbColor = Color(0xFF999999),
                            uncheckedTrackColor = Color(0xFF3a3a5a)
                        )
                    )
                }

                // Show Timer toggle (reduces anxiety for neurodivergent users)
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            text = "Show Timer",
                            fontWeight = FontWeight.Medium,
                            fontSize = 14.sp,
                            color = Color(0xFFE0E0E0)
                        )
                        Text(
                            text = "Hide to reduce time pressure",
                            fontSize = 12.sp,
                            color = Color(0xFF999999)
                        )
                    }
                    Switch(
                        checked = showTimer,
                        onCheckedChange = { onToggleShowTimer() },
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = Color(0xFFFFA500),
                            checkedTrackColor = Color(0xFFCC8400),
                            uncheckedThumbColor = Color(0xFF999999),
                            uncheckedTrackColor = Color(0xFF3a3a5a)
                        )
                    )
                }

                // Immersive Mode toggle (focus mode, auto-hide UI)
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            text = "Focus Mode",
                            fontWeight = FontWeight.Medium,
                            fontSize = 14.sp,
                            color = Color(0xFFE0E0E0)
                        )
                        Text(
                            text = "Auto-hide UI, tap to reveal",
                            fontSize = 12.sp,
                            color = Color(0xFF999999)
                        )
                    }
                    Switch(
                        checked = immersiveMode,
                        onCheckedChange = { onToggleImmersiveMode() },
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = Color(0xFF4CAF50),
                            checkedTrackColor = Color(0xFF388E3C),
                            uncheckedThumbColor = Color(0xFF999999),
                            uncheckedTrackColor = Color(0xFF3a3a5a)
                        )
                    )
                }

                HorizontalDivider(color = Color(0xFF4a4a6a), thickness = 1.dp)

                // Colorblind Mode selector (accessibility)
                var colorblindExpanded by remember { mutableStateOf(false) }
                Column {
                    Text(
                        text = "Color Vision",
                        fontWeight = FontWeight.Medium,
                        fontSize = 14.sp,
                        color = Color(0xFFE0E0E0),
                        modifier = Modifier.padding(bottom = 4.dp)
                    )
                    Box {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(8.dp))
                                .background(Color(0xFF3a3a5a))
                                .clickable { colorblindExpanded = true }
                                .padding(horizontal = 12.dp, vertical = 10.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column {
                                Text(
                                    text = colorblindMode.displayName,
                                    color = Color.White,
                                    fontSize = 14.sp
                                )
                                Text(
                                    text = colorblindMode.description,
                                    color = Color(0xFF999999),
                                    fontSize = 11.sp
                                )
                            }
                            Icon(
                                imageVector = Icons.Filled.KeyboardArrowDown,
                                contentDescription = "Select color mode",
                                tint = Color(0xFF7B68EE)
                            )
                        }
                        DropdownMenu(
                            expanded = colorblindExpanded,
                            onDismissRequest = { colorblindExpanded = false },
                            modifier = Modifier.background(Color(0xFF3a3a5a))
                        ) {
                            ColorblindMode.entries.forEach { mode ->
                                DropdownMenuItem(
                                    text = {
                                        Column {
                                            Text(
                                                text = mode.displayName,
                                                color = if (mode == colorblindMode) Color(0xFF7B68EE) else Color.White,
                                                fontWeight = if (mode == colorblindMode) FontWeight.Bold else FontWeight.Normal
                                            )
                                            Text(
                                                text = mode.description,
                                                color = Color(0xFF999999),
                                                fontSize = 11.sp
                                            )
                                        }
                                    },
                                    onClick = {
                                        onColorblindModeChange(mode)
                                        colorblindExpanded = false
                                    }
                                )
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("Done", color = Color(0xFF7B68EE))
            }
        }
    )
}

/**
 * Save/Load dialog showing 12 slots
 */
@Composable
private fun SaveLoadDialog(
    isSaveMode: Boolean,
    onFetchSlots: () -> List<SaveSlotInfo>,
    onSlotSelected: (Int) -> Unit,
    onSlotDeleted: (Int) -> Unit,
    onDismiss: () -> Unit
) {
    var showDeleteConfirm by remember { mutableStateOf<Int?>(null) }
    var refreshTrigger by remember { mutableStateOf(0) }

    // Fetch fresh slots each time trigger changes (e.g., after delete)
    val currentSlots = remember(refreshTrigger) { onFetchSlots() }

    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = Color(0xFF2a2a4a),
        titleContentColor = Color(0xFF7B68EE),
        textContentColor = Color(0xFFE0E0E0),
        title = {
            Text(
                text = if (isSaveMode) "Save Game" else "Load Game",
                fontWeight = FontWeight.Bold,
                fontSize = 20.sp
            )
        },
        text = {
            LazyColumn(
                modifier = Modifier.height(400.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(currentSlots) { slot ->
                    SaveSlotItem(
                        slot = slot,
                        isSaveMode = isSaveMode,
                        onClick = {
                            if (isSaveMode || !slot.isEmpty) {
                                onSlotSelected(slot.slotIndex)
                                onDismiss()
                            }
                        },
                        onDelete = {
                            if (!slot.isEmpty) {
                                showDeleteConfirm = slot.slotIndex
                            }
                        }
                    )
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel", color = Color(0xFF7B68EE))
            }
        }
    )

    // Delete confirmation dialog
    if (showDeleteConfirm != null) {
        AlertDialog(
            onDismissRequest = { showDeleteConfirm = null },
            containerColor = Color(0xFF2a2a4a),
            title = { Text("Delete Save?", color = Color(0xFF7B68EE)) },
            text = { Text("This cannot be undone.", color = Color(0xFFE0E0E0)) },
            confirmButton = {
                TextButton(onClick = {
                    onSlotDeleted(showDeleteConfirm!!)
                    refreshTrigger++
                    showDeleteConfirm = null
                }) {
                    Text("Delete", color = Color(0xFFF44336))
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteConfirm = null }) {
                    Text("Cancel", color = Color(0xFF7B68EE))
                }
            }
        )
    }
}

@Composable
private fun SaveSlotItem(
    slot: SaveSlotInfo,
    isSaveMode: Boolean,
    onClick: () -> Unit,
    onDelete: () -> Unit
) {
    val isClickable = isSaveMode || !slot.isEmpty

    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .clickable(enabled = isClickable) { onClick() },
        color = if (slot.isEmpty) Color(0xFF3a3a5a) else Color(0xFF4a4a6a),
        shape = RoundedCornerShape(8.dp)
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Slot ${slot.slotIndex + 1}",
                    fontWeight = FontWeight.Bold,
                    fontSize = 14.sp,
                    color = if (slot.isEmpty) Color(0xFF888888) else Color.White
                )
                if (!slot.isEmpty) {
                    Text(
                        text = "${slot.gridSize}x${slot.gridSize} ${slot.difficulty} - ${slot.formattedTime}",
                        fontSize = 12.sp,
                        color = Color(0xFFBBBBBB)
                    )
                    Text(
                        text = slot.formattedDate,
                        fontSize = 10.sp,
                        color = Color(0xFF888888)
                    )
                } else {
                    Text(
                        text = "Empty",
                        fontSize = 12.sp,
                        color = Color(0xFF666666)
                    )
                }
            }

            if (!slot.isEmpty) {
                IconButton(
                    onClick = onDelete,
                    modifier = Modifier.size(32.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Delete,
                        contentDescription = "Delete slot",
                        tint = Color(0xFFF44336),
                        modifier = Modifier.size(18.dp)
                    )
                }
            }
        }
    }
}
