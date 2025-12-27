/*
 * AccessibilityUtilsTest.kt: Tests for accessibility helper functions
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 */

package org.yegie.orthogon.ui

import org.junit.Assert.*
import org.junit.Test

class AccessibilityUtilsTest {

    @Test
    fun `formatTimeForSpeech with only seconds`() {
        assertEquals("0 seconds", formatTimeForSpeech(0))
        assertEquals("30 seconds", formatTimeForSpeech(30))
        assertEquals("59 seconds", formatTimeForSpeech(59))
    }

    @Test
    fun `formatTimeForSpeech with minutes only`() {
        assertEquals("1 minute", formatTimeForSpeech(60))
        assertEquals("2 minutes", formatTimeForSpeech(120))
        assertEquals("5 minutes", formatTimeForSpeech(300))
    }

    @Test
    fun `formatTimeForSpeech with minutes and seconds`() {
        assertEquals("1 minute and 30 seconds", formatTimeForSpeech(90))
        assertEquals("2 minutes and 15 seconds", formatTimeForSpeech(135))
        assertEquals("10 minutes and 5 seconds", formatTimeForSpeech(605))
    }

    @Test
    fun `buildVictoryAnnouncement formats correctly`() {
        val result = buildVictoryAnnouncement(5, "Easy", 125)
        assertTrue(result.contains("5 by 5"))
        assertTrue(result.contains("Easy"))
        assertTrue(result.contains("2 minutes"))
        assertTrue(result.contains("Congratulations"))
    }

    @Test
    fun `buildCellAccessibilityDescription for empty cell`() {
        val result = buildCellAccessibilityDescription(
            x = 0, y = 0,
            value = null,
            clue = "12+",
            isSelected = false,
            notes = listOf(false, false, false, false),
            puzzleSize = 4
        )
        assertTrue(result.contains("Row 1, column 1"))
        assertTrue(result.contains("empty"))
        assertTrue(result.contains("cage 12 plus"))
    }

    @Test
    fun `buildCellAccessibilityDescription for filled cell`() {
        val result = buildCellAccessibilityDescription(
            x = 2, y = 3,
            value = 5,
            clue = null,
            isSelected = true,
            notes = null,
            puzzleSize = 6
        )
        assertTrue(result.contains("Row 4, column 3"))
        assertTrue(result.contains("contains 5"))
        assertTrue(result.contains("selected"))
    }

    @Test
    fun `buildCellAccessibilityDescription with notes`() {
        val result = buildCellAccessibilityDescription(
            x = 1, y = 1,
            value = null,
            clue = null,
            isSelected = false,
            notes = listOf(true, false, true, false),
            puzzleSize = 4
        )
        assertTrue(result.contains("notes: 1, 3"))
    }

    @Test
    fun `buildCellAccessibilityDescription handles hex digits`() {
        val result = buildCellAccessibilityDescription(
            x = 0, y = 0,
            value = 12,
            clue = null,
            isSelected = false,
            notes = null,
            puzzleSize = 16
        )
        assertTrue(result.contains("C (12)"))
    }

    @Test
    fun `clue formatting expands operation symbols`() {
        // Testing via buildCellAccessibilityDescription
        val add = buildCellAccessibilityDescription(
            x = 0, y = 0, value = null, clue = "6+",
            isSelected = false, notes = null, puzzleSize = 4
        )
        assertTrue(add.contains("plus"))

        val minus = buildCellAccessibilityDescription(
            x = 0, y = 0, value = null, clue = "3-",
            isSelected = false, notes = null, puzzleSize = 4
        )
        assertTrue(minus.contains("minus"))

        val times = buildCellAccessibilityDescription(
            x = 0, y = 0, value = null, clue = "12×",
            isSelected = false, notes = null, puzzleSize = 4
        )
        assertTrue(times.contains("times"))

        val divide = buildCellAccessibilityDescription(
            x = 0, y = 0, value = null, clue = "2÷",
            isSelected = false, notes = null, puzzleSize = 4
        )
        assertTrue(divide.contains("divided by"))
    }

    @Test
    fun `numberButtonDescription formats correctly`() {
        assertEquals("Enter 5", numberButtonDescription(5, false))
        assertEquals("Add note 5", numberButtonDescription(5, true))
        assertEquals("Enter A (10)", numberButtonDescription(10, false))
    }
}
