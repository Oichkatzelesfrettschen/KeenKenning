/*
 * DesignTokens.kt: Typography, spacing, and cognitive load design tokens
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 *
 * Neurodivergent-friendly design considerations:
 * - Reduced visual clutter and cognitive load
 * - Clear visual hierarchy
 * - Consistent spacing rhythm
 * - Readable font sizes with good line height
 * - Minimal animations (respects reduced motion)
 * - Clear focus indicators
 */

package org.yegie.orthogon.ui.theme

import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Spacing scale based on 4px base unit.
 * Consistent rhythm reduces cognitive load.
 */
object Spacing {
    val xxs: Dp = 2.dp   // Minimal gaps
    val xs: Dp = 4.dp    // Tight spacing
    val sm: Dp = 8.dp    // Small gaps
    val md: Dp = 16.dp   // Standard spacing
    val lg: Dp = 24.dp   // Large gaps
    val xl: Dp = 32.dp   // Extra large
    val xxl: Dp = 48.dp  // Section spacing
    val xxxl: Dp = 64.dp // Major sections

    // Grid-specific spacing
    val cellPadding: Dp = 4.dp
    val cageBorderWidth: Dp = 2.dp
    val innerBorderWidth: Dp = 0.5.dp
    val gridPadding: Dp = 8.dp
}

/**
 * Typography scale for readability and hierarchy.
 * Uses system fonts for familiarity and performance.
 */
object Typography {
    // Display - for large puzzle digits
    val displayLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Bold,
        fontSize = 32.sp,
        lineHeight = 40.sp
    )

    val displayMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Bold,
        fontSize = 24.sp,
        lineHeight = 32.sp
    )

    // For cell digits (scales with cell size)
    val cellDigit = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Medium,
        fontSize = 20.sp,
        lineHeight = 24.sp
    )

    // For pencil marks (smaller)
    val pencilMark = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Normal,
        fontSize = 10.sp,
        lineHeight = 12.sp
    )

    // For cage operations (e.g., "12+")
    val cageOperation = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Medium,
        fontSize = 12.sp,
        lineHeight = 14.sp
    )

    // Headings
    val headlineLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Bold,
        fontSize = 24.sp,
        lineHeight = 32.sp
    )

    val headlineMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 20.sp,
        lineHeight = 28.sp
    )

    val headlineSmall = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 18.sp,
        lineHeight = 24.sp
    )

    // Body text
    val bodyLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Normal,
        fontSize = 16.sp,
        lineHeight = 24.sp
    )

    val bodyMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Normal,
        fontSize = 14.sp,
        lineHeight = 20.sp
    )

    val bodySmall = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Normal,
        fontSize = 12.sp,
        lineHeight = 16.sp
    )

    // Labels
    val labelLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Medium,
        fontSize = 14.sp,
        lineHeight = 20.sp
    )

    val labelMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Medium,
        fontSize = 12.sp,
        lineHeight = 16.sp
    )

    val labelSmall = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Medium,
        fontSize = 10.sp,
        lineHeight = 14.sp
    )
}

/**
 * Elevation scale for visual hierarchy.
 */
object Elevation {
    val none: Dp = 0.dp
    val low: Dp = 1.dp
    val medium: Dp = 4.dp
    val high: Dp = 8.dp
    val highest: Dp = 16.dp
}

/**
 * Corner radius scale for consistent rounding.
 */
object Corners {
    val none: Dp = 0.dp
    val xs: Dp = 2.dp
    val sm: Dp = 4.dp
    val md: Dp = 8.dp
    val lg: Dp = 12.dp
    val xl: Dp = 16.dp
    val full: Dp = 9999.dp  // Pill/circle
}

/**
 * Animation durations respecting reduced motion preferences.
 * Use AccessibilityManager to check for reduced motion setting.
 */
object AnimationDurations {
    const val instant: Int = 0
    const val fast: Int = 100
    const val normal: Int = 200
    const val slow: Int = 300
    const val verySlow: Int = 500
}

/**
 * Focus ring specifications for accessibility.
 */
object FocusRing {
    val width: Dp = 2.dp
    val offset: Dp = 2.dp
}

/**
 * Cognitive load reducers for neurodivergent users.
 */
object CognitiveAccessibility {
    /** Maximum simultaneous visual elements before grouping */
    const val maxVisibleDigits: Int = 9

    /** Hint display delay to reduce information overload */
    const val hintDelayMs: Long = 500

    /** Victory animation duration (short for reduced overwhelm) */
    const val celebrationDurationMs: Long = 1500

    /** Maximum pencil marks per cell to avoid visual noise */
    const val maxPencilMarksVisible: Int = 5

    /** Debounce time for rapid input protection */
    const val inputDebounceMs: Long = 100
}

/**
 * Battery saver mode for power optimization.
 * Works in conjunction with OLED mode (DisplayMode) from ColorSystem.kt.
 */
enum class BatterySaverMode {
    OFF,        // Full animations and effects
    MODERATE,   // Reduced animations, some effects disabled
    AGGRESSIVE  // Minimal animations, most effects disabled
}

/**
 * Battery saver settings for reducing power consumption.
 *
 * Power optimization strategies:
 * - Reduce animation durations/frames (less GPU work)
 * - Disable particle effects and complex transitions
 * - Lower refresh requirements for static content
 * - Combine with OLED mode for maximum savings on OLED displays
 */
object BatterySaver {
    /**
     * Animation duration multiplier for each mode.
     * Lower = faster animations = less GPU time.
     */
    fun getAnimationScale(mode: BatterySaverMode): Float = when (mode) {
        BatterySaverMode.OFF -> 1.0f
        BatterySaverMode.MODERATE -> 0.5f
        BatterySaverMode.AGGRESSIVE -> 0.0f // Instant, no animation
    }

    /**
     * Whether particle effects (AI visualization, victory confetti) are enabled.
     */
    fun areParticlesEnabled(mode: BatterySaverMode): Boolean = when (mode) {
        BatterySaverMode.OFF -> true
        BatterySaverMode.MODERATE -> false
        BatterySaverMode.AGGRESSIVE -> false
    }

    /**
     * Whether quantum/probability visualizations are enabled.
     */
    fun isQuantumVisualizationEnabled(mode: BatterySaverMode): Boolean = when (mode) {
        BatterySaverMode.OFF -> true
        BatterySaverMode.MODERATE -> true  // Keep for gameplay value
        BatterySaverMode.AGGRESSIVE -> false
    }

    /**
     * Whether smooth cell highlight transitions are enabled.
     */
    fun areSmoothTransitionsEnabled(mode: BatterySaverMode): Boolean = when (mode) {
        BatterySaverMode.OFF -> true
        BatterySaverMode.MODERATE -> true
        BatterySaverMode.AGGRESSIVE -> false // Instant state changes
    }

    /**
     * Victory celebration duration multiplier.
     */
    fun getCelebrationScale(mode: BatterySaverMode): Float = when (mode) {
        BatterySaverMode.OFF -> 1.0f
        BatterySaverMode.MODERATE -> 0.5f
        BatterySaverMode.AGGRESSIVE -> 0.2f // Very brief acknowledgment
    }

    /**
     * Helper to calculate scaled animation duration.
     */
    fun scaledDuration(baseDurationMs: Int, mode: BatterySaverMode): Int =
        (baseDurationMs * getAnimationScale(mode)).toInt()

    /**
     * Helper to calculate scaled celebration duration.
     */
    fun scaledCelebration(mode: BatterySaverMode): Long =
        (CognitiveAccessibility.celebrationDurationMs * getCelebrationScale(mode)).toLong()
}
