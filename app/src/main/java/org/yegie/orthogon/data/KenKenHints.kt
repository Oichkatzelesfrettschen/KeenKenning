/*
 * KenKenHints.kt: Step-by-step hint system wrapper
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 *
 * Provides Kotlin-friendly interface to native hint generation.
 * Hints explain logical deductions without giving away the full solution.
 */

package org.yegie.orthogon.data

/**
 * Hint types indicating what reasoning leads to the solution.
 */
enum class HintType(val code: Int) {
    NONE(0),           // No hint available
    NAKED_SINGLE(1),   // Only one value fits in this cell
    HIDDEN_SINGLE(2),  // Value can only go in this cell in row/column
    CAGE_FORCE(3),     // Cage arithmetic forces this value
    CAGE_SINGLE(4);    // Single-cell cage with given value

    companion object {
        fun fromCode(code: Int): HintType = values().find { it.code == code } ?: NONE
    }
}

/**
 * Result of a hint request.
 */
data class Hint(
    val type: HintType,
    val cell: Int,
    val row: Int,
    val col: Int,
    val value: Int,
    val cageRoot: Int,
    val relatedPosition: Int
) {
    /**
     * Generate a human-readable explanation of the hint.
     */
    fun explain(): String = when (type) {
        HintType.NONE -> "No hints available. The puzzle may require guessing."

        HintType.CAGE_SINGLE ->
            "Cell (${row + 1}, ${col + 1}) is a single-cell cage. The clue gives the answer directly."

        HintType.NAKED_SINGLE ->
            "Cell (${row + 1}, ${col + 1}) has only one possible value: $value. " +
            "All other values are eliminated by the row and column constraints."

        HintType.HIDDEN_SINGLE -> {
            val isRow = relatedPosition < 100
            val position = if (isRow) relatedPosition + 1 else relatedPosition - 100 + 1
            val dimension = if (isRow) "row" else "column"
            "In $dimension $position, the value $value can only go in cell (${row + 1}, ${col + 1})."
        }

        HintType.CAGE_FORCE ->
            "Based on the cage constraints, cell (${row + 1}, ${col + 1}) must be $value."
    }

    /**
     * Generate a progressive hint at the given level.
     * Level 1: Points to the region
     * Level 2: Points to the cell
     * Level 3: Gives the value
     */
    fun getProgressiveHint(level: Int): String = when {
        level <= 0 -> "Try looking at the grid more carefully."

        level == 1 -> when (type) {
            HintType.CAGE_SINGLE -> "Look at the single-cell cages."
            HintType.NAKED_SINGLE -> "Look at row ${row + 1} or column ${col + 1}."
            HintType.HIDDEN_SINGLE -> {
                val isRow = relatedPosition < 100
                val position = if (isRow) relatedPosition + 1 else relatedPosition - 100 + 1
                if (isRow) "Focus on row $position." else "Focus on column $position."
            }
            HintType.CAGE_FORCE -> "Examine the cage at (${row + 1}, ${col + 1})."
            HintType.NONE -> "No hints available."
        }

        level == 2 -> "Look at cell (${row + 1}, ${col + 1})."

        else -> "The value is $value."
    }
}

/**
 * JNI wrapper for native hint generation.
 */
object KenKenHints {

    init {
        System.loadLibrary("kenken-android-jni")
    }

    /**
     * Get the next available hint.
     */
    @JvmStatic
    external fun getNextHint(
        size: Int,
        grid: IntArray,
        dsf: IntArray,
        clues: LongArray,
        solution: IntArray?,
        modeFlags: Int
    ): IntArray?

    /**
     * Explain a specific cell.
     */
    @JvmStatic
    external fun explainCell(
        size: Int,
        cell: Int,
        grid: IntArray,
        dsf: IntArray,
        clues: LongArray,
        solution: IntArray?,
        modeFlags: Int
    ): IntArray?

    /**
     * Get a hint with Kotlin-friendly return type.
     */
    fun getHint(
        size: Int,
        grid: IntArray,
        dsf: IntArray,
        clues: LongArray,
        solution: IntArray? = null,
        modeFlags: Int = 0
    ): Hint? {
        val result = getNextHint(size, grid, dsf, clues, solution, modeFlags) ?: return null
        return parseHintResult(result)
    }

    /**
     * Explain why a cell has a particular value.
     */
    fun explain(
        size: Int,
        cell: Int,
        grid: IntArray,
        dsf: IntArray,
        clues: LongArray,
        solution: IntArray? = null,
        modeFlags: Int = 0
    ): Hint? {
        val result = explainCell(size, cell, grid, dsf, clues, solution, modeFlags) ?: return null
        return parseHintResult(result)
    }

    private fun parseHintResult(data: IntArray): Hint {
        require(data.size == 7) { "Invalid hint result size: ${data.size}" }
        return Hint(
            type = HintType.fromCode(data[0]),
            cell = data[1],
            row = data[2],
            col = data[3],
            value = data[4],
            cageRoot = data[5],
            relatedPosition = data[6]
        )
    }
}
