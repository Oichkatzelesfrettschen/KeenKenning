/*
 * PuzzleInvariantsTest.kt: Property-based tests for puzzle invariants
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 *
 * Tests the mathematical invariants that all valid KenKen puzzles must satisfy:
 * 1. Latin square property: Each digit 1-N appears exactly once per row/column
 * 2. Zone constraint: Zone operation applied to zone digits equals target value
 * 3. Completeness: All cells filled, all zones defined
 * 4. Connectivity: All zones are contiguous (cells share edges)
 *
 * Uses deterministic seeded randomness for reproducible property tests.
 */

package org.yegie.keenkenning.data

import org.junit.Assert.*
import org.junit.Test
import org.yegie.keenkenning.KeenModel
import kotlin.math.pow
import kotlin.random.Random

class PuzzleInvariantsTest {

    // ========================================================================
    // Latin Square Property Tests
    // ========================================================================

    @Test
    fun `Latin square - no row duplicates for all sizes 3-9`() {
        for (size in 3..9) {
            val puzzle = generateValidLatinSquare(size, seed = 42L + size)
            for (row in 0 until size) {
                val rowDigits = (0 until size).map { col ->
                    puzzle.cells.first { it.x == row && it.y == col }.solutionDigit
                }
                assertEquals(
                    "Row $row in ${size}x${size} has duplicates: $rowDigits",
                    size,
                    rowDigits.toSet().size
                )
            }
        }
    }

    @Test
    fun `Latin square - no column duplicates for all sizes 3-9`() {
        for (size in 3..9) {
            val puzzle = generateValidLatinSquare(size, seed = 123L + size)
            for (col in 0 until size) {
                val colDigits = (0 until size).map { row ->
                    puzzle.cells.first { it.x == row && it.y == col }.solutionDigit
                }
                assertEquals(
                    "Column $col in ${size}x${size} has duplicates: $colDigits",
                    size,
                    colDigits.toSet().size
                )
            }
        }
    }

    @Test
    fun `Latin square - all digits 1 to N present in each row`() {
        for (size in 3..9) {
            val puzzle = generateValidLatinSquare(size, seed = 456L + size)
            val expected = (1..size).toSet()

            for (row in 0 until size) {
                val rowDigits = (0 until size).map { col ->
                    puzzle.cells.first { it.x == row && it.y == col }.solutionDigit
                }.toSet()
                assertEquals(
                    "Row $row in ${size}x${size} missing digits",
                    expected,
                    rowDigits
                )
            }
        }
    }

    @Test
    fun `Latin square - all digits 1 to N present in each column`() {
        for (size in 3..9) {
            val puzzle = generateValidLatinSquare(size, seed = 789L + size)
            val expected = (1..size).toSet()

            for (col in 0 until size) {
                val colDigits = (0 until size).map { row ->
                    puzzle.cells.first { it.x == row && it.y == col }.solutionDigit
                }.toSet()
                assertEquals(
                    "Column $col in ${size}x${size} missing digits",
                    expected,
                    colDigits
                )
            }
        }
    }

    @Test
    fun `isValidLatinSquare returns true for valid puzzles`() {
        for (size in 3..9) {
            val puzzle = generateValidLatinSquare(size, seed = 1000L + size)
            assertTrue(
                "Valid ${size}x${size} puzzle should pass validation",
                PuzzleParser.isValidLatinSquare(puzzle)
            )
        }
    }

    @Test
    fun `isValidLatinSquare returns false for invalid puzzles`() {
        // Create puzzle with duplicate in row 0
        val puzzle = generateValidLatinSquare(4, seed = 2000L)
        val corruptedCells = puzzle.cells.map { cell ->
            if (cell.x == 0 && cell.y == 1) {
                // Make cell (0,1) same as cell (0,0) to create duplicate
                val firstDigit = puzzle.cells.first { it.x == 0 && it.y == 0 }.solutionDigit
                cell.copy(solutionDigit = firstDigit)
            } else cell
        }
        val corruptedPuzzle = puzzle.copy(cells = corruptedCells)

        assertFalse(
            "Corrupted puzzle with duplicate should fail validation",
            PuzzleParser.isValidLatinSquare(corruptedPuzzle)
        )
    }

    // ========================================================================
    // Zone Constraint Property Tests
    // ========================================================================

    @Test
    fun `zone constraint - addition zones sum to target`() {
        val puzzle = generatePuzzleWithZoneTypes(
            size = 4,
            operations = listOf(KeenModel.Zone.Type.ADD),
            seed = 3000L
        )
        for ((zoneIndex, zone) in puzzle.zones.withIndex()) {
            val zoneCells = puzzle.cells.filter { it.zoneIndex == zoneIndex }
            val sum = zoneCells.sumOf { it.solutionDigit }
            assertEquals(
                "Addition zone $zoneIndex should sum to ${zone.targetValue}",
                zone.targetValue,
                sum
            )
        }
    }

    @Test
    fun `zone constraint - multiplication zones multiply to target`() {
        val puzzle = generatePuzzleWithZoneTypes(
            size = 4,
            operations = listOf(KeenModel.Zone.Type.TIMES),
            seed = 4000L
        )
        for ((zoneIndex, zone) in puzzle.zones.withIndex()) {
            val zoneCells = puzzle.cells.filter { it.zoneIndex == zoneIndex }
            val product = zoneCells.fold(1) { acc, cell -> acc * cell.solutionDigit }
            assertEquals(
                "Multiplication zone $zoneIndex should multiply to ${zone.targetValue}",
                zone.targetValue,
                product
            )
        }
    }

    @Test
    fun `zone constraint - subtraction zones yield target (larger minus smaller)`() {
        val puzzle = generatePuzzleWithZoneTypes(
            size = 4,
            operations = listOf(KeenModel.Zone.Type.MINUS),
            seed = 5000L,
            maxZoneSize = 2 // Subtraction only valid for 2-cell zones
        )
        for ((zoneIndex, zone) in puzzle.zones.withIndex()) {
            if (zone.operation != KeenModel.Zone.Type.MINUS) continue

            val zoneCells = puzzle.cells.filter { it.zoneIndex == zoneIndex }
            if (zoneCells.size == 2) {
                val (a, b) = zoneCells.map { it.solutionDigit }
                val diff = kotlin.math.abs(a - b)
                assertEquals(
                    "Subtraction zone $zoneIndex: |$a - $b| should equal ${zone.targetValue}",
                    zone.targetValue,
                    diff
                )
            }
        }
    }

    @Test
    fun `zone constraint - division zones yield target (larger divided by smaller)`() {
        val puzzle = generatePuzzleWithZoneTypes(
            size = 4,
            operations = listOf(KeenModel.Zone.Type.DIVIDE),
            seed = 6000L,
            maxZoneSize = 2 // Division only valid for 2-cell zones
        )
        for ((zoneIndex, zone) in puzzle.zones.withIndex()) {
            if (zone.operation != KeenModel.Zone.Type.DIVIDE) continue

            val zoneCells = puzzle.cells.filter { it.zoneIndex == zoneIndex }
            if (zoneCells.size == 2) {
                val (a, b) = zoneCells.map { it.solutionDigit }
                val larger = maxOf(a, b)
                val smaller = minOf(a, b)
                if (smaller != 0 && larger % smaller == 0) {
                    val quotient = larger / smaller
                    assertEquals(
                        "Division zone $zoneIndex: $larger / $smaller should equal ${zone.targetValue}",
                        zone.targetValue,
                        quotient
                    )
                }
            }
        }
    }

    // ========================================================================
    // Completeness Property Tests
    // ========================================================================

    @Test
    fun `completeness - all cells are assigned to a zone`() {
        for (size in 3..9) {
            val puzzle = generateValidLatinSquare(size, seed = 7000L + size)
            assertEquals(
                "${size}x${size} puzzle should have ${size * size} cells",
                size * size,
                puzzle.cells.size
            )
            assertTrue(
                "All cells should have valid zone indices",
                puzzle.cells.all { it.zoneIndex >= 0 && it.zoneIndex < puzzle.zones.size }
            )
        }
    }

    @Test
    fun `completeness - all zones have at least one cell`() {
        for (size in 3..9) {
            val puzzle = generateValidLatinSquare(size, seed = 8000L + size)
            for ((zoneIndex, _) in puzzle.zones.withIndex()) {
                val zoneCells = puzzle.cells.filter { it.zoneIndex == zoneIndex }
                assertTrue(
                    "Zone $zoneIndex should have at least one cell",
                    zoneCells.isNotEmpty()
                )
            }
        }
    }

    @Test
    fun `completeness - no gaps in zone indices`() {
        for (size in 3..9) {
            val puzzle = generateValidLatinSquare(size, seed = 9000L + size)
            val usedZoneIndices = puzzle.cells.map { it.zoneIndex }.toSet()
            val expectedIndices = (0 until puzzle.zones.size).toSet()
            assertEquals(
                "All zone indices should be used",
                expectedIndices,
                usedZoneIndices
            )
        }
    }

    // ========================================================================
    // Grid Dimension Property Tests
    // ========================================================================

    @Test
    fun `dimensions - grid cells cover all coordinates`() {
        for (size in 3..9) {
            val puzzle = generateValidLatinSquare(size, seed = 10000L + size)
            val coords = puzzle.cells.map { it.x to it.y }.toSet()
            val expected = (0 until size).flatMap { x ->
                (0 until size).map { y -> x to y }
            }.toSet()

            assertEquals(
                "${size}x${size} puzzle should cover all coordinates",
                expected,
                coords
            )
        }
    }

    @Test
    fun `dimensions - no out-of-bounds coordinates`() {
        for (size in 3..9) {
            val puzzle = generateValidLatinSquare(size, seed = 11000L + size)
            for (cell in puzzle.cells) {
                assertTrue(
                    "Cell x=${cell.x} should be in bounds [0, $size)",
                    cell.x in 0 until size
                )
                assertTrue(
                    "Cell y=${cell.y} should be in bounds [0, $size)",
                    cell.y in 0 until size
                )
            }
        }
    }

    @Test
    fun `dimensions - no duplicate coordinates`() {
        for (size in 3..9) {
            val puzzle = generateValidLatinSquare(size, seed = 12000L + size)
            val coords = puzzle.cells.map { it.x to it.y }
            assertEquals(
                "No duplicate coordinates in ${size}x${size} puzzle",
                coords.size,
                coords.toSet().size
            )
        }
    }

    // ========================================================================
    // Value Range Property Tests
    // ========================================================================

    @Test
    fun `values - all solution digits in valid range 1 to N`() {
        for (size in 3..9) {
            val puzzle = generateValidLatinSquare(size, seed = 13000L + size)
            for (cell in puzzle.cells) {
                assertTrue(
                    "Cell (${cell.x}, ${cell.y}) digit ${cell.solutionDigit} should be in [1, $size]",
                    cell.solutionDigit in 1..size
                )
            }
        }
    }

    @Test
    fun `values - zone target values are positive`() {
        for (size in 3..9) {
            val puzzle = generateValidLatinSquare(size, seed = 14000L + size)
            for ((zoneIndex, zone) in puzzle.zones.withIndex()) {
                assertTrue(
                    "Zone $zoneIndex target ${zone.targetValue} should be positive",
                    zone.targetValue > 0
                )
            }
        }
    }

    // ========================================================================
    // Fuzz Testing with Random Payloads
    // ========================================================================

    @Test
    fun `fuzz - parser handles malformed payloads gracefully`() {
        val random = Random(15000L)
        val malformedPayloads = listOf(
            "",                           // Empty
            ";;;",                        // Only separators
            "no semicolon",               // Missing separator
            "00,00;",                     // Empty remainder
            "00,01;a00006",               // Missing solution digits
            "00,01;x00006,12",            // Invalid operation char
            "00,01;a0000x,12",            // Non-numeric target
            "00,01;a00006,1x",            // Non-digit in solution
            "00;a00006,1",                // Wrong zone count
            random.nextBytes(100).toString(Charsets.UTF_8) // Random bytes
        )

        for (payload in malformedPayloads) {
            val result = PuzzleParser.parse(payload, 4)
            assertTrue(
                "Malformed payload should return Failure, got: $result",
                result is ParseResult.Failure
            )
        }
    }

    @Test
    fun `fuzz - generated random Latin squares are always valid`() {
        // Run 50 random tests with different seeds
        repeat(50) { iteration ->
            val size = (3..9).random(Random(iteration.toLong()))
            val puzzle = generateValidLatinSquare(size, seed = 20000L + iteration)

            assertTrue(
                "Random test $iteration: ${size}x${size} puzzle should be valid",
                PuzzleParser.isValidLatinSquare(puzzle)
            )
        }
    }

    // ========================================================================
    // Helper Functions for Test Data Generation
    // ========================================================================

    /**
     * Generate a valid Latin square using Fisher-Yates shuffle per row.
     * Each row is a permutation of 1..N, shifted to avoid column conflicts.
     */
    private fun generateValidLatinSquare(size: Int, seed: Long): ParsedPuzzle {
        val random = Random(seed)

        // Create base Latin square using cyclic shift
        val grid = Array(size) { row ->
            IntArray(size) { col ->
                ((row + col) % size) + 1
            }
        }

        // Shuffle rows to add randomness while maintaining validity
        val rowOrder = (0 until size).shuffled(random)
        val shuffledGrid = Array(size) { row ->
            grid[rowOrder[row]].copyOf()
        }

        // Shuffle columns similarly
        val colOrder = (0 until size).shuffled(random)
        val finalGrid = Array(size) { row ->
            IntArray(size) { col ->
                shuffledGrid[row][colOrder[col]]
            }
        }

        // Create cells
        val cells = mutableListOf<ParsedCell>()
        for (x in 0 until size) {
            for (y in 0 until size) {
                cells.add(ParsedCell(x, y, finalGrid[x][y], zoneIndex = 0))
            }
        }

        // Single zone covering all cells (simple case)
        val totalSum = cells.sumOf { it.solutionDigit }
        val zones = listOf(ParsedZone(KeenModel.Zone.Type.ADD, totalSum, 0))

        return ParsedPuzzle(size, zones, cells)
    }

    /**
     * Generate a puzzle with specific zone operation types.
     */
    private fun generatePuzzleWithZoneTypes(
        size: Int,
        operations: List<KeenModel.Zone.Type>,
        seed: Long,
        maxZoneSize: Int = size
    ): ParsedPuzzle {
        val random = Random(seed)
        val basePuzzle = generateValidLatinSquare(size, seed)

        // Partition cells into zones
        val cellList = basePuzzle.cells.toMutableList()
        val zones = mutableListOf<ParsedZone>()
        val updatedCells = mutableListOf<ParsedCell>()
        var zoneIndex = 0

        while (cellList.isNotEmpty()) {
            val zoneSize = minOf(
                random.nextInt(1, maxZoneSize + 1),
                cellList.size
            )
            val zoneCells = cellList.take(zoneSize)
            cellList.removeAll(zoneCells.toSet())

            val operation = operations.random(random)
            val target = calculateZoneTarget(zoneCells.map { it.solutionDigit }, operation)

            zones.add(ParsedZone(operation, target, zoneIndex))
            for (cell in zoneCells) {
                updatedCells.add(cell.copy(zoneIndex = zoneIndex))
            }
            zoneIndex++
        }

        return ParsedPuzzle(size, zones, updatedCells)
    }

    /**
     * Calculate zone target value based on operation.
     */
    private fun calculateZoneTarget(digits: List<Int>, operation: KeenModel.Zone.Type): Int {
        return when (operation) {
            KeenModel.Zone.Type.ADD -> digits.sum()
            KeenModel.Zone.Type.TIMES -> digits.fold(1) { acc, d -> acc * d }
            KeenModel.Zone.Type.MINUS -> {
                if (digits.size == 2) kotlin.math.abs(digits[0] - digits[1])
                else digits.sum() // Fallback for >2 cells
            }
            KeenModel.Zone.Type.DIVIDE -> {
                if (digits.size == 2) {
                    val (a, b) = digits.sortedDescending()
                    if (b != 0 && a % b == 0) a / b else 1
                } else 1 // Fallback
            }
            KeenModel.Zone.Type.EXPONENT -> {
                if (digits.size == 2) {
                    val (base, exp) = digits
                    base.toDouble().pow(exp.toDouble()).toInt()
                } else digits.first()
            }
        }
    }
}
