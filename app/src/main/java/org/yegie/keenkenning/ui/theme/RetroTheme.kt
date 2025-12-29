/*
 * RetroTheme.kt: 8-bit retro theme with voxel-style visuals
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 *
 * Uses Press Start 2P font (SIL OFL 1.1):
 * Copyright 2012 The Press Start 2P Project Authors
 * https://fonts.google.com/specimen/Press+Start+2P
 */

package org.yegie.keenkenning.ui.theme

import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.yegie.keenkenning.R

/**
 * Press Start 2P font family for 8-bit aesthetics.
 * Best rendered at multiples of 8px (8, 16, 24, 32...).
 */
val PressStart2P = FontFamily(
    Font(R.font.press_start_2p, FontWeight.Normal)
)

/**
 * 8-bit color palette inspired by NES/Famicom.
 * Limited palette creates authentic retro feel.
 */
object RetroPalette {
    // Background colors (dark to light gradient)
    val black = Color(0xFF0F0F1B)       // Near-black with slight blue
    val darkGray = Color(0xFF1E1E2E)    // Dark background
    val midGray = Color(0xFF3A3A5A)     // Mid tone
    val lightGray = Color(0xFF7A7A9A)   // Light accents

    // Primary colors (NES-inspired)
    val red = Color(0xFFE04040)         // Bright red
    val orange = Color(0xFFE07820)      // Orange
    val yellow = Color(0xFFF0D020)      // Bright yellow
    val green = Color(0xFF40C040)       // Classic green
    val cyan = Color(0xFF40C0C0)        // Cyan/teal
    val blue = Color(0xFF4080E0)        // Bright blue
    val purple = Color(0xFF8040C0)      // Purple
    val magenta = Color(0xFFE040C0)     // Magenta/pink

    // UI colors
    val white = Color(0xFFF0F0F0)       // Off-white (easier on eyes)
    val cream = Color(0xFFF8F8E8)       // Cream for text backgrounds
    val highlight = Color(0xFFFFFF80)   // Yellow highlight
    val shadow = Color(0xFF000000)      // Pure black for shadows

    // Zone colors for cages (high saturation, distinct hues)
    val zone1 = Color(0xFFE04040)   // Red
    val zone2 = Color(0xFF40C040)   // Green
    val zone3 = Color(0xFF4080E0)   // Blue
    val zone4 = Color(0xFFF0D020)   // Yellow
    val zone5 = Color(0xFF8040C0)   // Purple
    val zone6 = Color(0xFFE07820)   // Orange
    val zone7 = Color(0xFF40C0C0)   // Cyan
    val zone8 = Color(0xFFE040C0)   // Magenta
    val zone9 = Color(0xFFA0A0A0)   // Gray

    fun zoneColor(index: Int): Color {
        val colors = listOf(zone1, zone2, zone3, zone4, zone5, zone6, zone7, zone8, zone9)
        return colors[index % colors.size]
    }

    /**
     * Zone background colors (darker variants for cell backgrounds).
     */
    fun zoneBackground(index: Int): Color {
        return zoneColor(index).copy(alpha = 0.25f)
    }
}

/**
 * 8-bit game colors with voxel/pixel art aesthetic.
 */
val Retro8BitColors = GameColors(
    surface = RetroPalette.darkGray,
    surfaceDim = RetroPalette.black,
    primary = RetroPalette.cyan,
    primaryContainer = RetroPalette.midGray,
    onPrimary = RetroPalette.black,
    selected = RetroPalette.highlight,
    selectedBorder = RetroPalette.yellow,
    textPrimary = RetroPalette.white,
    textSecondary = RetroPalette.lightGray,
    textHint = RetroPalette.midGray,
    borderThick = RetroPalette.white,
    borderThin = RetroPalette.midGray,
    success = RetroPalette.green,
    error = RetroPalette.red,
    warning = RetroPalette.orange,
    noteText = RetroPalette.cyan,
    hintText = RetroPalette.purple,
    buttonPrimary = RetroPalette.blue,
    buttonSecondary = RetroPalette.midGray,
    buttonText = RetroPalette.white
)

/**
 * 8-bit dimensions - chunkier, more voxel-like.
 * Uses pixel-perfect sizes (multiples of 4dp).
 */
val Retro8BitDimensions = GameDimensions(
    cellMinSize = 44.dp,          // Slightly larger for chunky look
    cellPadding = 4.dp,           // 4dp grid alignment

    // Thick borders for voxel effect
    cageBorderWidth = 4.dp,       // Chunky cage borders
    gridBorderWidth = 2.dp,       // Visible grid lines

    // 8-bit friendly text sizes (multiples of 8sp)
    clueTextSize = 8.sp,          // Small but readable
    valueTextSize = 16.sp,        // Main digit size
    noteTextSize = 8.sp,          // Pencil marks
    buttonTextSize = 16.sp,       // Button labels

    // Touch targets
    buttonMinSize = 48.dp,
    touchTargetMin = 48.dp,

    // Spacing
    gridPadding = 8.dp,
    controlSpacing = 8.dp,

    // Voxel-style clue box
    clueBoxElevation = 4.dp,      // More pronounced shadow
    clueBoxCornerRadius = 0.dp,   // Sharp corners for pixel look
    clueBoxPaddingHorizontal = 4.dp,
    clueBoxPaddingVertical = 2.dp,

    // Positioning ratios
    valueVerticalOffsetRatio = 0.12f,
    noteBoxSizeRatio = 0.30f,
    noteGridOffsetRatio = 0.18f,
    hintGridOffsetRatio = 0.20f
)

/**
 * 8-bit typography using Press Start 2P font.
 */
object RetroTypography {
    val displayLarge = TextStyle(
        fontFamily = PressStart2P,
        fontWeight = FontWeight.Normal,
        fontSize = 24.sp,
        lineHeight = 32.sp
    )

    val displayMedium = TextStyle(
        fontFamily = PressStart2P,
        fontWeight = FontWeight.Normal,
        fontSize = 16.sp,
        lineHeight = 24.sp
    )

    val cellDigit = TextStyle(
        fontFamily = PressStart2P,
        fontWeight = FontWeight.Normal,
        fontSize = 16.sp,
        lineHeight = 20.sp
    )

    val pencilMark = TextStyle(
        fontFamily = PressStart2P,
        fontWeight = FontWeight.Normal,
        fontSize = 8.sp,
        lineHeight = 10.sp
    )

    val cageOperation = TextStyle(
        fontFamily = PressStart2P,
        fontWeight = FontWeight.Normal,
        fontSize = 8.sp,
        lineHeight = 10.sp
    )

    val button = TextStyle(
        fontFamily = PressStart2P,
        fontWeight = FontWeight.Normal,
        fontSize = 12.sp,
        lineHeight = 16.sp
    )

    val label = TextStyle(
        fontFamily = PressStart2P,
        fontWeight = FontWeight.Normal,
        fontSize = 8.sp,
        lineHeight = 12.sp
    )
}

/**
 * Whether 8-bit retro mode is active.
 */
val LocalRetroMode = staticCompositionLocalOf { false }

/**
 * 8-bit Retro Theme provider.
 */
@Composable
fun RetroGameTheme(
    enabled: Boolean = true,
    content: @Composable () -> Unit
) {
    val colors = if (enabled) Retro8BitColors else GameColors()
    val dimensions = if (enabled) Retro8BitDimensions else GameDimensions()

    CompositionLocalProvider(
        LocalGameColors provides colors,
        LocalGameDimensions provides dimensions,
        LocalRetroMode provides enabled
    ) {
        content()
    }
}

/**
 * Extension to get zone color for 8-bit mode.
 */
fun ZoneColors.forRetroMode(zoneId: Int, isRetro: Boolean): Color {
    return if (isRetro) {
        RetroPalette.zoneBackground(zoneId)
    } else {
        forZone(zoneId, highContrast = false)
    }
}

/**
 * Voxel-style button dimensions.
 */
object VoxelButton {
    val borderWidth: Dp = 4.dp
    val shadowOffset: Dp = 4.dp
    val pressedOffset: Dp = 2.dp
    val cornerRadius: Dp = 0.dp  // Sharp corners for pixel look
}
