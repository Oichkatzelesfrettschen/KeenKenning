/*
 * ColorSystem.kt: Colorblind-friendly design system tokens
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 *
 * Implements the Indigo-Gray-Magenta palette from the Colorblindness design system.
 * Provides variants for: Default, Protan, Deutan, Tritan, Mono (achromatopsia).
 *
 * Design principles:
 * - Never rely on color alone (use shapes, patterns, icons, underlines)
 * - Maintain contrast: 4.5:1 text, 3:1 UI components, 7:1 critical
 * - Separate by lightness, not just hue
 * - Limit simultaneous accents (1 primary, 1 accent, neutrals)
 */

package org.yegie.orthogon.ui.theme

import androidx.compose.ui.graphics.Color

/**
 * Color Vision Deficiency profile for adaptive palettes.
 */
enum class CvdProfile {
    DEFAULT,
    PROTAN,   // Red-weak
    DEUTAN,   // Green-weak
    TRITAN,   // Blue-weak
    MONO      // Achromatopsia
}

/**
 * Brand color tokens for a specific CVD profile.
 */
data class BrandColors(
    val primary: Color,
    val primaryStrong: Color,
    val accent: Color,
    val accentStrong: Color,
    val text: Color,
    val surface: Color,
    val border: Color,
    val focusRing: Color,
    val link: Color
)

/**
 * Semantic color tokens for game-specific elements with CVD support.
 */
data class CvdGameColors(
    val cellBackground: Color,
    val cellBackgroundSelected: Color,
    val cellBackgroundError: Color,
    val cellBackgroundCorrect: Color,
    val cageBorder: Color,
    val cageBorderHighlight: Color,
    val digitNormal: Color,
    val digitPencilMark: Color,
    val digitHint: Color,
    val operationSymbol: Color,
    // Cage clue box (elevated surface with depth)
    val clueBoxBackground: Color,
    val clueBoxText: Color,
    // Zone category colors (with shape/pattern fallback)
    val zoneColors: List<Color>
)

/**
 * Display mode for power optimization.
 */
enum class DisplayMode {
    STANDARD,  // Normal dark/light mode
    OLED       // Pure black backgrounds for OLED power savings
}

/**
 * Gray scale ramp consistent across all profiles.
 */
object GrayScale {
    val gray900 = Color(0xFF111827)
    val gray800 = Color(0xFF1F2937)
    val gray700 = Color(0xFF374151)
    val gray500 = Color(0xFF6B7280)
    val gray400 = Color(0xFF9CA3AF)
    val gray300 = Color(0xFFD1D5DB)
    val gray200 = Color(0xFFE5E7EB)
    val gray100 = Color(0xFFF3F4F6)
    val gray0 = Color(0xFFFFFFFF)
}

/**
 * OLED-optimized colors with true black for power savings.
 * On OLED displays, #000000 pixels are completely OFF.
 */
object OledColors {
    val pureBlack = Color(0xFF000000)
    val nearBlack = Color(0xFF0A0A0A)
    val darkSurface = Color(0xFF121212)
    val darkBorder = Color(0xFF252525)
}

/**
 * Indigo scale for default profile.
 */
private object IndigoDefault {
    val i700 = Color(0xFF3730A3)
    val i600 = Color(0xFF4F46E5)
    val i500 = Color(0xFF6366F1)
    val i300 = Color(0xFFA5B4FC)
    val i100 = Color(0xFFE0E7FF)
}

/**
 * Magenta scale for default profile.
 */
private object MagentaDefault {
    val m700 = Color(0xFF86198F)
    val m600 = Color(0xFFC026D3)
    val m400 = Color(0xFFE879F9)
    val m200 = Color(0xFFF5D0FE)
}

/**
 * Default profile brand colors.
 */
val DefaultBrand = BrandColors(
    primary = IndigoDefault.i600,
    primaryStrong = IndigoDefault.i700,
    accent = MagentaDefault.m600,
    accentStrong = MagentaDefault.m700,
    text = GrayScale.gray900,
    surface = GrayScale.gray0,
    border = GrayScale.gray300,
    focusRing = IndigoDefault.i300,
    link = IndigoDefault.i700
)

/**
 * Protan (red-weak) profile - darkened Indigo, warmer/brighter Magenta.
 */
val ProtanBrand = BrandColors(
    primary = Color(0xFF3D35C8),
    primaryStrong = Color(0xFF2E2AA1),
    accent = Color(0xFFD62A8A),
    accentStrong = Color(0xFFB12179),
    text = GrayScale.gray900,
    surface = GrayScale.gray0,
    border = GrayScale.gray300,
    focusRing = IndigoDefault.i300,
    link = Color(0xFF2E2AA1)
)

/**
 * Deutan (green-weak) profile - similar adjustments to Protan.
 */
val DeutanBrand = BrandColors(
    primary = Color(0xFF3F38C5),
    primaryStrong = Color(0xFF2E2AA1),
    accent = Color(0xFFD02C8C),
    accentStrong = Color(0xFFA81E7B),
    text = GrayScale.gray900,
    surface = GrayScale.gray0,
    border = GrayScale.gray300,
    focusRing = IndigoDefault.i300,
    link = Color(0xFF2E2AA1)
)

/**
 * Tritan (blue-weak) profile - violet-shifted Indigo, redder Magenta.
 */
val TritanBrand = BrandColors(
    primary = Color(0xFF6940D6),
    primaryStrong = Color(0xFF5B33BF),
    accent = Color(0xFFC81A78),
    accentStrong = Color(0xFFA9156E),
    text = GrayScale.gray900,
    surface = GrayScale.gray0,
    border = GrayScale.gray300,
    focusRing = IndigoDefault.i300,
    link = Color(0xFF5B33BF)
)

/**
 * Mono (achromatopsia) profile - grayscale only, rely on patterns/shapes.
 */
val MonoBrand = BrandColors(
    primary = Color(0xFF334155),
    primaryStrong = Color(0xFF1E293B),
    accent = Color(0xFF6B7280),
    accentStrong = Color(0xFF4B5563),
    text = GrayScale.gray900,
    surface = GrayScale.gray0,
    border = GrayScale.gray300,
    focusRing = GrayScale.gray400,
    link = Color(0xFF1E293B)
)

/**
 * Get brand colors for a specific CVD profile.
 */
fun getBrandColors(profile: CvdProfile): BrandColors = when (profile) {
    CvdProfile.DEFAULT -> DefaultBrand
    CvdProfile.PROTAN -> ProtanBrand
    CvdProfile.DEUTAN -> DeutanBrand
    CvdProfile.TRITAN -> TritanBrand
    CvdProfile.MONO -> MonoBrand
}

/**
 * Get game-specific colors for a CVD profile.
 * Zone colors use brand categorical palette with shape fallback.
 *
 * @param profile Color vision deficiency profile
 * @param isDarkMode Whether dark mode is enabled
 * @param displayMode Display optimization mode (STANDARD or OLED)
 */
fun getCvdGameColors(
    profile: CvdProfile,
    isDarkMode: Boolean,
    displayMode: DisplayMode = DisplayMode.STANDARD
): CvdGameColors {
    val brand = getBrandColors(profile)
    val isOled = displayMode == DisplayMode.OLED

    // Zone colors - limited to 6 categories as per design guide
    // Uses markers (circle, triangle, square, diamond) as fallback
    val zoneColors = when (profile) {
        CvdProfile.DEFAULT -> listOf(
            Color(0xFF3730A3), Color(0xFFC026D3), Color(0xFF6366F1),
            Color(0xFF374151), Color(0xFFA5B4FC), Color(0xFFE879F9)
        )
        CvdProfile.PROTAN -> listOf(
            Color(0xFF2E2AA1), Color(0xFFD62A8A), Color(0xFF5A5FEF),
            Color(0xFF374151), Color(0xFFA5B4FC), Color(0xFFEC6FB1)
        )
        CvdProfile.DEUTAN -> listOf(
            Color(0xFF2E2AA1), Color(0xFFD02C8C), Color(0xFF5D60EE),
            Color(0xFF374151), Color(0xFFA5B4FC), Color(0xFFF07BC0)
        )
        CvdProfile.TRITAN -> listOf(
            Color(0xFF5B33BF), Color(0xFFC81A78), Color(0xFF7E5AEE),
            Color(0xFF374151), Color(0xFFA5B4FC), Color(0xFFE05C9F)
        )
        CvdProfile.MONO -> listOf(
            Color(0xFF1E293B), Color(0xFF6B7280), Color(0xFF475569),
            Color(0xFF374151), Color(0xFF9CA3AF), Color(0xFFD1D5DB)
        )
    }

    return when {
        // OLED mode: Pure black backgrounds for maximum power savings
        isOled && isDarkMode -> CvdGameColors(
            cellBackground = OledColors.pureBlack,
            cellBackgroundSelected = brand.primaryStrong,
            cellBackgroundError = Color(0xFF5C1010), // Darker red for OLED
            cellBackgroundCorrect = Color(0xFF0D3D1F), // Darker green for OLED
            cageBorder = OledColors.darkBorder,
            cageBorderHighlight = brand.accent,
            digitNormal = GrayScale.gray100,
            digitPencilMark = GrayScale.gray400,
            digitHint = brand.accent,
            operationSymbol = GrayScale.gray300,
            clueBoxBackground = OledColors.darkSurface,  // Subtle elevation on OLED
            clueBoxText = GrayScale.gray100,
            zoneColors = zoneColors
        )
        // Standard dark mode
        isDarkMode -> CvdGameColors(
            cellBackground = GrayScale.gray800,
            cellBackgroundSelected = brand.primaryStrong,
            cellBackgroundError = Color(0xFF7F1D1D), // Dark red, with icon
            cellBackgroundCorrect = Color(0xFF14532D), // Dark green, with icon
            cageBorder = GrayScale.gray500,
            cageBorderHighlight = brand.accent,
            digitNormal = GrayScale.gray100,
            digitPencilMark = GrayScale.gray400,
            digitHint = brand.accent,
            operationSymbol = GrayScale.gray300,
            clueBoxBackground = GrayScale.gray700,  // Slightly lighter than cell for depth
            clueBoxText = GrayScale.gray100,
            zoneColors = zoneColors
        )
        // Light mode (OLED setting ignored in light mode)
        else -> CvdGameColors(
            cellBackground = GrayScale.gray0,
            cellBackgroundSelected = brand.focusRing,
            cellBackgroundError = Color(0xFFFEE2E2), // Light red, with icon
            cellBackgroundCorrect = Color(0xFFDCFCE7), // Light green, with icon
            cageBorder = GrayScale.gray700,
            cageBorderHighlight = brand.primary,
            digitNormal = GrayScale.gray900,
            digitPencilMark = GrayScale.gray500,
            digitHint = brand.accent,
            operationSymbol = GrayScale.gray700,
            clueBoxBackground = Color(0xFFFAFAFA),  // Slightly off-white for subtle contrast
            clueBoxText = GrayScale.gray900,
            zoneColors = zoneColors
        )
    }
}

/**
 * Zone marker shapes for non-color differentiation.
 * Use alongside colors for CVD accessibility.
 */
enum class ZoneMarker {
    CIRCLE,     // Zone 0
    TRIANGLE,   // Zone 1
    SQUARE,     // Zone 2
    DIAMOND,    // Zone 3
    CROSS,      // Zone 4
    PLUS        // Zone 5
}

/**
 * Get marker shape for a zone index.
 */
fun getZoneMarker(zoneIndex: Int): ZoneMarker =
    ZoneMarker.entries[zoneIndex % ZoneMarker.entries.size]
