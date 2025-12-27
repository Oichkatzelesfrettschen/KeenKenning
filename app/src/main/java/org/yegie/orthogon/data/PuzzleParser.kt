/*
 * PuzzleParser.kt: Pure Kotlin parsing of JNI puzzle payloads
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 *
 * Extracts parsing logic from KenKenModelBuilder for testability.
 * JNI payload format: "<zone_indices>;<zone_defs><solution_digits>"
 *
 * Example 3x3: "00,01,01,02,02,02,03,03,03;a00006,m00002,a00003,s00001,123456789"
 */

package org.yegie.orthogon.data

import org.yegie.orthogon.KenKenModel

/**
 * Parsed zone definition from JNI payload.
 */
data class ParsedZone(
    val operation: KenKenModel.Zone.Type,
    val targetValue: Int,
    val index: Int
)

/**
 * Parsed cell with zone assignment and solution digit.
 */
data class ParsedCell(
    val x: Int,
    val y: Int,
    val solutionDigit: Int,
    val zoneIndex: Int
)

/**
 * Complete parsed puzzle structure.
 */
data class ParsedPuzzle(
    val size: Int,
    val zones: List<ParsedZone>,
    val cells: List<ParsedCell>
)

/**
 * Result of parsing a JNI payload.
 */
sealed interface ParseResult {
    data class Success(val puzzle: ParsedPuzzle) : ParseResult
    data class Failure(val message: String, val position: Int? = null) : ParseResult
}

/**
 * Pure Kotlin parser for JNI puzzle payloads.
 * No JNI dependencies - fully testable in JVM unit tests.
 */
object PuzzleParser {

    /**
     * Parse a JNI payload string into a ParsedPuzzle.
     *
     * @param payload The raw string from JNI (format: "zones;operations|digits")
     * @param size Expected grid size
     * @return ParseResult with puzzle or error details
     */
    fun parse(payload: String, size: Int): ParseResult {
        if (payload.isBlank()) {
            return ParseResult.Failure("Empty payload")
        }

        // Split by semicolon: first part is zone indices, rest is zone defs + solution
        val semicolonIndex = payload.indexOf(';')
        if (semicolonIndex == -1) {
            return ParseResult.Failure("Missing semicolon separator", 0)
        }

        val zoneIndicesRaw = payload.substring(0, semicolonIndex)
        val remainder = payload.substring(semicolonIndex + 1)

        // Parse zone indices (2-digit numbers separated by commas or directly concatenated)
        val zoneIndices = parseZoneIndices(zoneIndicesRaw, size)
        if (zoneIndices.size != size * size) {
            return ParseResult.Failure(
                "Zone indices count mismatch: expected ${size * size}, got ${zoneIndices.size}",
                0
            )
        }

        // Count unique zones
        val uniqueZones = zoneIndices.toSet().size

        // Parse zone definitions (7 chars each: operation char + 5-digit value)
        val zoneDefs = mutableListOf<ParsedZone>()
        var pos = 0
        for (i in 0 until uniqueZones) {
            if (pos + 6 > remainder.length) {
                return ParseResult.Failure("Truncated zone definition at zone $i", semicolonIndex + 1 + pos)
            }

            val opChar = remainder[pos]
            val valueStr = remainder.substring(pos + 1, pos + 6)
            val value = valueStr.toIntOrNull()
                ?: return ParseResult.Failure("Invalid zone value: $valueStr", semicolonIndex + 1 + pos + 1)

            val operation = when (opChar) {
                'a' -> KenKenModel.Zone.Type.ADD
                'm' -> KenKenModel.Zone.Type.TIMES
                's' -> KenKenModel.Zone.Type.MINUS
                'd' -> KenKenModel.Zone.Type.DIVIDE
                'e' -> KenKenModel.Zone.Type.EXPONENT
                else -> return ParseResult.Failure("Unknown operation: $opChar", semicolonIndex + 1 + pos)
            }

            zoneDefs.add(ParsedZone(operation, value, i))
            pos += 6
            // Skip comma if present
            if (pos < remainder.length && remainder[pos] == ',') {
                pos++
            }
        }

        // Parse solution digits (remaining characters)
        val solutionPart = remainder.substring(pos)
        if (solutionPart.length < size * size) {
            return ParseResult.Failure(
                "Solution digits count mismatch: expected ${size * size}, got ${solutionPart.length}",
                semicolonIndex + 1 + pos
            )
        }

        // Build cells with zone mapping
        val cells = mutableListOf<ParsedCell>()
        val zoneMapping = buildZoneMapping(zoneIndices)

        for (i in 0 until size * size) {
            val digitChar = solutionPart[i]
            val digit = digitChar.digitToIntOrNull()
                ?: return ParseResult.Failure("Invalid digit: $digitChar", semicolonIndex + 1 + pos + i)

            val x = i / size
            val y = i % size
            val rawZoneId = zoneIndices[i]
            val mappedZoneIndex = zoneMapping[rawZoneId]
                ?: return ParseResult.Failure("Unmapped zone: $rawZoneId", 0)

            cells.add(ParsedCell(x, y, digit, mappedZoneIndex))
        }

        return ParseResult.Success(
            ParsedPuzzle(size, zoneDefs, cells)
        )
    }

    /**
     * Parse zone indices from the header section.
     * Format: "00,01,01,02,..." or "00010102..." (2 digits per cell)
     */
    @Suppress("UNUSED_PARAMETER")  // size reserved for future validation
    private fun parseZoneIndices(raw: String, size: Int): List<Int> {
        val indices = mutableListOf<Int>()

        // Try comma-separated first
        if (raw.contains(',')) {
            raw.split(',').forEach { part ->
                part.trim().toIntOrNull()?.let { indices.add(it) }
            }
        } else {
            // Assume 2-digit concatenated
            var i = 0
            while (i + 1 < raw.length) {
                raw.substring(i, i + 2).toIntOrNull()?.let { indices.add(it) }
                i += 2
            }
        }

        return indices
    }

    /**
     * Build mapping from raw zone IDs to sequential indices.
     */
    private fun buildZoneMapping(zoneIndices: List<Int>): Map<Int, Int> {
        val mapping = mutableMapOf<Int, Int>()
        var nextIndex = 0

        for (rawId in zoneIndices) {
            if (rawId !in mapping) {
                mapping[rawId] = nextIndex++
            }
        }

        return mapping
    }

    /**
     * Validate that a puzzle forms a valid Latin square.
     */
    fun isValidLatinSquare(puzzle: ParsedPuzzle): Boolean {
        val size = puzzle.size
        val grid = Array(size) { IntArray(size) }

        // Fill grid
        for (cell in puzzle.cells) {
            grid[cell.x][cell.y] = cell.solutionDigit
        }

        // Check rows
        for (row in 0 until size) {
            val seen = mutableSetOf<Int>()
            for (col in 0 until size) {
                if (!seen.add(grid[row][col])) return false
            }
        }

        // Check columns
        for (col in 0 until size) {
            val seen = mutableSetOf<Int>()
            for (row in 0 until size) {
                if (!seen.add(grid[row][col])) return false
            }
        }

        return true
    }
}
