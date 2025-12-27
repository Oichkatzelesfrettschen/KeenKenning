/*
 * PuzzleParserTest.kt: Unit tests for puzzle payload parsing
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 */

package org.yegie.orthogon.data

import org.junit.Assert.*
import org.junit.Test
import org.yegie.orthogon.KenKenModel

class PuzzleParserTest {

    @Test
    fun `parse empty payload returns failure`() {
        val result = PuzzleParser.parse("", 3)
        assertTrue(result is ParseResult.Failure)
        assertEquals("Empty payload", (result as ParseResult.Failure).message)
    }

    @Test
    fun `parse payload without semicolon returns failure`() {
        val result = PuzzleParser.parse("no semicolon here", 3)
        assertTrue(result is ParseResult.Failure)
        assertEquals("Missing semicolon separator", (result as ParseResult.Failure).message)
    }

    @Test
    fun `parse valid 3x3 puzzle`() {
        // Simulated 3x3 payload: 4 zones, solution 1-9
        // Zone layout: 00,00,01,00,02,01,03,02,03
        // Zones: a=6 (1+2+3), m=2 (1*2), a=9 (4+5), s=1 (3-2)
        val payload = "00,00,01,00,02,01,03,02,03;a00006,m00002,a00009,s00001,123456789"

        val result = PuzzleParser.parse(payload, 3)
        assertTrue("Expected Success, got $result", result is ParseResult.Success)

        val puzzle = (result as ParseResult.Success).puzzle
        assertEquals(3, puzzle.size)
        assertEquals(4, puzzle.zones.size)
        assertEquals(9, puzzle.cells.size)

        // Verify zone operations
        assertEquals(KenKenModel.Zone.Type.ADD, puzzle.zones[0].operation)
        assertEquals(6, puzzle.zones[0].targetValue)
        assertEquals(KenKenModel.Zone.Type.TIMES, puzzle.zones[1].operation)
        assertEquals(2, puzzle.zones[1].targetValue)
    }

    @Test
    fun `parse valid 4x4 puzzle with all operations`() {
        // 4x4 with 5 zones using different operations
        val zoneIndices = "00,00,01,01,00,02,02,01,03,03,04,04,03,03,04,04"
        val zoneDefs = "a00010,m00024,s00002,d00002,e00008"
        val solution = "1234234134124321"
        val payload = "$zoneIndices;$zoneDefs,$solution"

        val result = PuzzleParser.parse(payload, 4)
        assertTrue("Expected Success, got $result", result is ParseResult.Success)

        val puzzle = (result as ParseResult.Success).puzzle
        assertEquals(4, puzzle.size)
        assertEquals(5, puzzle.zones.size)

        // Verify all operation types present
        val ops = puzzle.zones.map { it.operation }.toSet()
        assertTrue(KenKenModel.Zone.Type.ADD in ops)
        assertTrue(KenKenModel.Zone.Type.TIMES in ops)
        assertTrue(KenKenModel.Zone.Type.MINUS in ops)
        assertTrue(KenKenModel.Zone.Type.DIVIDE in ops)
        assertTrue(KenKenModel.Zone.Type.EXPONENT in ops)
    }

    @Test
    fun `parse with concatenated zone indices`() {
        // Format without commas: "000001010203..." - 9 cells, 5 unique zones
        val zoneIndices = "000001010203030404"
        // 5 zone definitions for 5 unique zones (00,01,02,03,04)
        val zoneDefs = "a00006,m00004,a00008,s00002,d00003,"
        val solution = "123456789"
        val payload = "$zoneIndices;$zoneDefs$solution"

        val result = PuzzleParser.parse(payload, 3)
        assertTrue("Expected Success, got $result", result is ParseResult.Success)
    }

    @Test
    fun `zone count mismatch returns failure`() {
        // Only 8 zone indices for a 3x3 (should be 9)
        val payload = "00,00,01,01,02,02,03,03;a00006,123456789"
        val result = PuzzleParser.parse(payload, 3)
        assertTrue(result is ParseResult.Failure)
        assertTrue((result as ParseResult.Failure).message.contains("mismatch"))
    }

    @Test
    fun `invalid operation character returns failure`() {
        val payload = "00,00,01,01,02,02,03,03,03;x00006,123456789"
        val result = PuzzleParser.parse(payload, 3)
        assertTrue(result is ParseResult.Failure)
        assertTrue((result as ParseResult.Failure).message.contains("Unknown operation"))
    }

    @Test
    fun `invalid digit in solution returns failure`() {
        // 4 unique zones (00,01,02,03), need 4 zone defs, then solution with X
        val payload = "00,00,01,01,02,02,03,03,03;a00006,m00002,a00003,s00001,12345678X"
        val result = PuzzleParser.parse(payload, 3)
        assertTrue("Expected Failure, got $result", result is ParseResult.Failure)
        assertTrue((result as ParseResult.Failure).message.contains("Invalid digit"))
    }

    @Test
    fun `isValidLatinSquare validates correctly`() {
        // Valid 3x3 Latin square
        val validPuzzle = ParsedPuzzle(
            size = 3,
            zones = listOf(ParsedZone(KenKenModel.Zone.Type.ADD, 6, 0)),
            cells = listOf(
                ParsedCell(0, 0, 1, 0), ParsedCell(0, 1, 2, 0), ParsedCell(0, 2, 3, 0),
                ParsedCell(1, 0, 2, 0), ParsedCell(1, 1, 3, 0), ParsedCell(1, 2, 1, 0),
                ParsedCell(2, 0, 3, 0), ParsedCell(2, 1, 1, 0), ParsedCell(2, 2, 2, 0)
            )
        )
        assertTrue(PuzzleParser.isValidLatinSquare(validPuzzle))

        // Invalid: duplicate in row
        val invalidPuzzle = validPuzzle.copy(
            cells = validPuzzle.cells.toMutableList().apply {
                this[1] = this[1].copy(solutionDigit = 1) // Now row 0 has two 1s
            }
        )
        assertFalse(PuzzleParser.isValidLatinSquare(invalidPuzzle))
    }

    @Test
    fun `cell coordinates are correctly assigned`() {
        val payload = "00,01,02,03,04,05,06,07,08;a00001,a00002,a00003,a00004,a00005,a00006,a00007,a00008,a00009,123456789"
        val result = PuzzleParser.parse(payload, 3)
        assertTrue(result is ParseResult.Success)

        val cells = (result as ParseResult.Success).puzzle.cells

        // Verify coordinate assignment: i/size = x, i%size = y
        assertEquals(0, cells[0].x)  // i=0: 0/3=0
        assertEquals(0, cells[0].y)  // i=0: 0%3=0
        assertEquals(0, cells[1].x)  // i=1: 1/3=0
        assertEquals(1, cells[1].y)  // i=1: 1%3=1
        assertEquals(1, cells[3].x)  // i=3: 3/3=1
        assertEquals(0, cells[3].y)  // i=3: 3%3=0
    }

    @Test
    fun `large zone values parsed correctly`() {
        // Test 5-digit zone values
        val payload = "00,00,00,00,00,00,00,00,00;a99999,123456789"
        val result = PuzzleParser.parse(payload, 3)
        assertTrue(result is ParseResult.Success)

        val zone = (result as ParseResult.Success).puzzle.zones[0]
        assertEquals(99999, zone.targetValue)
    }
}
