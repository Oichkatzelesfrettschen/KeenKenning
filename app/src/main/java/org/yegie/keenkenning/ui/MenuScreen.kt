/*
 * MenuScreen.kt: Compose UI for the main menu
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning.ui

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.yegie.keenkenning.data.GameMode
import org.yegie.keenkenning.data.GridSize
import org.yegie.keenkenning.data.Difficulty

data class MenuState(
    val selectedSize: Int = 5,
    val selectedDifficulty: Int = 1,
    val selectedMode: GameMode = GameMode.STANDARD,
    val canContinue: Boolean = false,
    // Legacy compatibility
    val multiplicationOnly: Boolean = false
)

@Suppress("UNUSED_PARAMETER")  // onMultiplicationToggle kept for legacy API compatibility
@Composable
fun MenuScreen(
    state: MenuState,
    onSizeChange: (Int) -> Unit,
    onDifficultyChange: (Int) -> Unit,
    onModeChange: (GameMode) -> Unit = {},
    onMultiplicationToggle: (Boolean) -> Unit = {},
    onStartGame: () -> Unit,
    onContinueGame: () -> Unit
) {
    val difficulties = Difficulty.entries.map { it.displayName }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    colors = listOf(
                        Color(0xFF1a1a2e),
                        Color(0xFF16213e)
                    )
                )
            )
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 24.dp, vertical = 16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.height(24.dp))

            // Title - marked as heading for screen reader navigation
            Text(
                text = "KEEN",
                fontSize = 42.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF7B68EE),
                letterSpacing = 4.sp,
                modifier = Modifier.semantics { heading() }
            )

            Text(
                text = "Classic Logic Puzzle",
                fontSize = 14.sp,
                color = Color(0xFF888888),
                letterSpacing = 2.sp
            )

            Spacer(modifier = Modifier.height(20.dp))

            // Game Mode Selector (Scrolling Cards)
            GameModeSelector(
                selectedMode = state.selectedMode,
                onModeChange = onModeChange
            )

            Spacer(modifier = Modifier.height(16.dp))

            // Grid Size Selector
            val sizes = GridSize.allSizes()
            SelectorRow(
                label = "Grid Size",
                options = sizes.map { it.displayName },
                selectedIndex = sizes.indexOfFirst { it.size == state.selectedSize }.coerceAtLeast(0),
                onSelect = { index -> onSizeChange(sizes[index].size) },
                testTag = "gridSizeSelector"
            )

            Spacer(modifier = Modifier.height(12.dp))

            // Difficulty Selector
            SelectorRow(
                label = "Difficulty",
                options = difficulties,
                selectedIndex = state.selectedDifficulty,
                onSelect = onDifficultyChange,
                testTag = "difficultySelector"
            )

            Spacer(modifier = Modifier.weight(1f))

            // Start Button
            Button(
                onClick = onStartGame,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(64.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF00CED1)
                ),
                shape = RoundedCornerShape(16.dp),
                elevation = ButtonDefaults.buttonElevation(
                    defaultElevation = 8.dp,
                    pressedElevation = 4.dp
                )
            ) {
                Text(
                    text = "START GAME",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Black,
                    letterSpacing = 3.sp,
                    color = Color(0xFF1a1a2e)
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Continue Button
            if (state.canContinue) {
                OutlinedButton(
                    onClick = onContinueGame,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(48.dp),
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = Color(0xFF7B68EE)
                    ),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Text(
                        text = "CONTINUE",
                        fontSize = 16.sp,
                        letterSpacing = 2.sp
                    )
                }
            }

            Spacer(modifier = Modifier.height(12.dp))
        }
    }
}

/**
 * Horizontal scrolling game mode selector with cards.
 */
@Composable
fun GameModeSelector(
    selectedMode: GameMode,
    onModeChange: (GameMode) -> Unit
) {
    val listState = rememberLazyListState()

    // Track scroll position to show/hide fade indicators
    val canScrollLeft by remember {
        derivedStateOf { listState.canScrollBackward }
    }
    val canScrollRight by remember {
        derivedStateOf { listState.canScrollForward }
    }

    Column(modifier = Modifier.fillMaxWidth()) {
        Text(
            text = "Game Mode",
            fontSize = 12.sp,
            color = Color(0xFF888888),
            modifier = Modifier.padding(bottom = 8.dp)
        )

        // Wrap LazyRow in a Box with fade edge indicators for scroll affordance
        // Note: Using fixed height because LazyRow (SubcomposeLayout) doesn't support IntrinsicSize
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(110.dp)  // ModeCard is 100.dp + scale animation allowance
        ) {
            val availableModes = remember { GameMode.availableModes() }
        LazyRow(
            state = listState,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            contentPadding = PaddingValues(horizontal = 4.dp),
            modifier = Modifier.testTag("gameModeList")
        ) {
                items(availableModes) { mode ->
                    ModeCard(
                        mode = mode,
                        isSelected = mode == selectedMode,
                        onClick = {
                            if (mode.implemented) {
                                onModeChange(mode)
                            }
                        }
                    )
                }
            }

            // Left fade edge - "glass river" effect with visible depth shadow
            if (canScrollLeft) {
                Box(
                    modifier = Modifier
                        .align(Alignment.CenterStart)
                        .width(48.dp)
                        .fillMaxHeight()
                        .background(
                            Brush.horizontalGradient(
                                colors = listOf(
                                    Color(0xFF0d0d18),          // Deeper shadow for depth
                                    Color(0xFF1a1a2e),          // Match background
                                    Color(0xFF1a1a2e).copy(alpha = 0.5f),
                                    Color.Transparent
                                )
                            )
                        )
                )
            }

            // Right fade edge - "glass river" effect with visible depth shadow
            if (canScrollRight) {
                Box(
                    modifier = Modifier
                        .align(Alignment.CenterEnd)
                        .width(48.dp)
                        .fillMaxHeight()
                        .background(
                            Brush.horizontalGradient(
                                colors = listOf(
                                    Color.Transparent,
                                    Color(0xFF1a1a2e).copy(alpha = 0.5f),
                                    Color(0xFF1a1a2e),          // Match background
                                    Color(0xFF0d0d18)           // Deeper shadow for depth
                                )
                            )
                        )
                )
            }
        }

        // Selected mode description
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = selectedMode.description,
            fontSize = 13.sp,
            color = Color(0xFFAAAAAA),
            modifier = Modifier.padding(horizontal = 4.dp)
        )

        // Extended tip (if available) - shown with info icon
        selectedMode.extendedTip?.let { tip ->
            Spacer(modifier = Modifier.height(6.dp))
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 4.dp)
                    .background(
                        color = Color(0xFF2a2a4a).copy(alpha = 0.5f),
                        shape = RoundedCornerShape(8.dp)
                    )
                    .padding(8.dp),
                verticalAlignment = Alignment.Top
            ) {
                Icon(
                    imageVector = Icons.Default.Info,
                    contentDescription = "Tip",
                    tint = Color(0xFF7B68EE),
                    modifier = Modifier.size(16.dp)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = tip,
                    fontSize = 12.sp,
                    color = Color(0xFF9999AA),
                    lineHeight = 16.sp
                )
            }
        }
    }
}

/**
 * Individual mode card in the horizontal selector.
 */
@Composable
fun ModeCard(
    mode: GameMode,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    val scale by animateFloatAsState(
        targetValue = if (isSelected) 1.05f else 1f,
        animationSpec = tween(150),
        label = "cardScale"
    )

    val backgroundColor by animateColorAsState(
        targetValue = when {
            isSelected -> Color(0xFF7B68EE)
            mode.implemented -> Color(0xFF2a2a4a)
            else -> Color(0xFF1a1a2a)
        },
        animationSpec = tween(200),
        label = "cardBg"
    )

    val borderColor by animateColorAsState(
        targetValue = when {
            isSelected -> Color(0xFF9B88FF)
            mode.implemented -> Color(0xFF3a3a5a)
            else -> Color(0xFF2a2a3a)
        },
        animationSpec = tween(200),
        label = "cardBorder"
    )

    // Build accessible description for mode card
    val accessibilityDescription = buildString {
        append(mode.displayName)
        append(" mode")
        if (isSelected) append(", selected")
        if (!mode.implemented) append(", coming soon")
    }

    Card(
        modifier = Modifier
            .width(90.dp)
            .height(100.dp)
            .scale(scale)
            .clickable(enabled = mode.implemented, onClick = onClick)
            .semantics {
                contentDescription = accessibilityDescription
                if (isSelected) {
                    stateDescription = "selected"
                }
            },
        colors = CardDefaults.cardColors(containerColor = backgroundColor),
        shape = RoundedCornerShape(12.dp),
        border = androidx.compose.foundation.BorderStroke(
            width = if (isSelected) 2.dp else 1.dp,
            color = borderColor
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(8.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            // Icon
            Icon(
                imageVector = getModeIcon(mode),
                contentDescription = mode.displayName,
                tint = when {
                    isSelected -> Color.White
                    mode.implemented -> Color(0xFF888888)
                    else -> Color(0xFF555555)
                },
                modifier = Modifier.size(28.dp)
            )

            Spacer(modifier = Modifier.height(6.dp))

            // Name
            Text(
                text = mode.displayName,
                fontSize = 11.sp,
                fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                color = when {
                    isSelected -> Color.White
                    mode.implemented -> Color(0xFFCCCCCC)
                    else -> Color(0xFF666666)
                },
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                textAlign = TextAlign.Center
            )

            // Coming Soon badge for unimplemented modes
            if (!mode.implemented) {
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = "Soon",
                    fontSize = 8.sp,
                    color = Color(0xFF666666),
                    fontWeight = FontWeight.Light
                )
            }
        }
    }
}

/**
 * Map mode to Material icon.
 * Uses only icons available in the standard Material Icons library.
 */
@Composable
fun getModeIcon(mode: GameMode): ImageVector {
    // Using only icons from material-icons-core (no extended library needed)
    return when (mode) {
        GameMode.STANDARD -> Icons.Default.Add           // Standard: all operations (+)
        GameMode.MULTIPLICATION_ONLY -> Icons.Default.Close  // Multiply: × symbol
        GameMode.MYSTERY -> Icons.Default.Info           // Mystery: info/question
        GameMode.ZERO_INCLUSIVE -> Icons.Default.Check   // Zero: included (check)
        GameMode.EXPONENT -> Icons.Default.Star          // Power: special star
        GameMode.NEGATIVE_NUMBERS -> Icons.Default.Delete    // Negative: minus-like
        GameMode.BITWISE -> Icons.Default.Settings       // Bitwise: XOR operations
        GameMode.MODULAR -> Icons.Default.Refresh        // Modular: wrap around
        GameMode.KILLER -> Icons.Default.Warning         // Killer: constraint warning
        GameMode.HINT_MODE -> Icons.Default.Info         // Hints: info
        GameMode.ADAPTIVE -> Icons.Default.Settings      // Adaptive: settings/tune
        GameMode.STORY -> Icons.Default.Menu             // Story: list/book-like
        GameMode.RETRO_8BIT -> Icons.Default.PlayArrow   // 8-Bit: retro game controller
    }
}

@Composable
fun SelectorRow(
    label: String,
    options: List<String>,
    selectedIndex: Int,
    onSelect: (Int) -> Unit,
    testTag: String? = null
) {
    var expanded by remember { mutableStateOf(false) }

    Column(modifier = Modifier.fillMaxWidth()) {
        Text(
            text = label,
            fontSize = 12.sp,
            color = Color(0xFF888888),
            modifier = Modifier.padding(bottom = 4.dp)
        )

        Box {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .background(Color(0xFF2a2a4a))
                    .clickable { expanded = true }
                    .padding(16.dp)
                    .let { base -> if (testTag != null) base.testTag(testTag) else base },
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = options.getOrElse(selectedIndex) { options.first() },
                    color = Color.White,
                    fontSize = 16.sp
                )
                Icon(
                    imageVector = Icons.Default.KeyboardArrowDown,
                    contentDescription = "Expand",
                    tint = Color(0xFF7B68EE)
                )
            }

            DropdownMenu(
                expanded = expanded,
                onDismissRequest = { expanded = false },
                modifier = Modifier.background(Color(0xFF2a2a4a))
            ) {
                options.forEachIndexed { index, option ->
                    val itemModifier = if (testTag != null) {
                        Modifier.testTag("${testTag}Option_$index")
                    } else {
                        Modifier
                    }
                    DropdownMenuItem(
                        modifier = itemModifier,
                        text = {
                            Text(
                                text = option,
                                color = if (index == selectedIndex) Color(0xFF7B68EE) else Color.White
                            )
                        },
                        onClick = {
                            onSelect(index)
                            expanded = false
                        }
                    )
                }
            }
        }
    }
}

@Composable
fun ToggleOption(
    label: String,
    subtitle: String? = null,
    checked: Boolean,
    onToggle: (Boolean) -> Unit,
    accentColor: Color = Color(0xFF7B68EE)
) {
    val backgroundColor by animateColorAsState(
        targetValue = if (checked) accentColor.copy(alpha = 0.2f) else Color.Transparent,
        animationSpec = tween(200),
        label = "toggleBg"
    )

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(backgroundColor)
            .border(
                width = 1.dp,
                color = if (checked) accentColor else Color(0xFF3a3a5a),
                shape = RoundedCornerShape(8.dp)
            )
            .clickable { onToggle(!checked) }
            .padding(horizontal = 16.dp, vertical = 12.dp)
            .semantics {
                stateDescription = if (checked) "on" else "off"
                contentDescription = "$label, ${if (checked) "on" else "off"}, double tap to toggle"
            },
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column {
            Text(
                text = label,
                color = if (checked) accentColor else Color(0xFFCCCCCC),
                fontSize = 16.sp
            )
            if (subtitle != null) {
                Text(
                    text = subtitle,
                    color = Color(0xFF666666),
                    fontSize = 11.sp
                )
            }
        }

        Switch(
            checked = checked,
            onCheckedChange = onToggle,
            colors = SwitchDefaults.colors(
                checkedThumbColor = accentColor,
                checkedTrackColor = accentColor.copy(alpha = 0.5f),
                uncheckedThumbColor = Color(0xFF666666),
                uncheckedTrackColor = Color(0xFF3a3a5a)
            )
        )
    }
}
