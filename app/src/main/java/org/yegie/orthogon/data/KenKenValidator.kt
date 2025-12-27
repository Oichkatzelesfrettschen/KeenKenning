/*
 * KenKenValidator.kt: Real-time validation wrapper for native validation
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 *
 * Provides Kotlin-friendly interface to native validation functions.
 * Used by GameViewModel for real-time error highlighting.
 */

package org.yegie.orthogon.data

/**
 * Error flags returned by validation.
 * Can be combined with bitwise OR.
 */
object ValidationError {
    const val OK = 0x00        // No errors
    const val ROW = 0x01       // Duplicate in row
    const val COLUMN = 0x02   // Duplicate in column
    const val CAGE = 0x04     // Cage constraint violated

    fun hasRowError(flags: Int) = (flags and ROW) != 0
    fun hasColumnError(flags: Int) = (flags and COLUMN) != 0
    fun hasCageError(flags: Int) = (flags and CAGE) != 0
    fun hasAnyError(flags: Int) = flags != OK
}

/**
 * JNI wrapper for native KenKen validation.
 * All methods are static and thread-safe.
 */
object KenKenValidator {

    init {
        System.loadLibrary("kenken-android-jni")
    }

    /**
     * Validate entire grid and return error flags for each cell.
     *
     * @param size Grid dimension (NxN)
     * @param grid Current cell values (0 = empty), row-major order
     * @param dsf DSF array for cage membership
     * @param clues Cage clues with operation encoded in upper bits
     * @param modeFlags Mode flags (e.g., MODE_KILLER)
     * @return IntArray of error flags for each cell
     */
    @JvmStatic
    external fun validateGrid(
        size: Int,
        grid: IntArray,
        dsf: IntArray,
        clues: LongArray,
        modeFlags: Int
    ): IntArray?

    /**
     * Check if puzzle is complete and valid.
     *
     * @return 1 if complete and valid, 0 otherwise
     */
    @JvmStatic
    external fun isComplete(
        size: Int,
        grid: IntArray,
        dsf: IntArray,
        clues: LongArray,
        modeFlags: Int
    ): Int

    /**
     * Kotlin-friendly validation that returns a typed result.
     */
    fun validate(
        size: Int,
        grid: IntArray,
        dsf: IntArray,
        clues: LongArray,
        modeFlags: Int = 0
    ): ValidationResult {
        val errors = validateGrid(size, grid, dsf, clues, modeFlags)
            ?: return ValidationResult(emptyList(), 0)

        val errorCells = errors.mapIndexed { index, flags ->
            CellError(
                row = index / size,
                col = index % size,
                flags = flags
            )
        }.filter { it.flags != ValidationError.OK }

        return ValidationResult(errorCells, errors.count { it != ValidationError.OK })
    }

    /**
     * Check completion with Kotlin-friendly return type.
     */
    fun checkComplete(
        size: Int,
        grid: IntArray,
        dsf: IntArray,
        clues: LongArray,
        modeFlags: Int = 0
    ): Boolean = isComplete(size, grid, dsf, clues, modeFlags) == 1
}

/**
 * Error information for a single cell.
 */
data class CellError(
    val row: Int,
    val col: Int,
    val flags: Int
) {
    val hasRowError get() = ValidationError.hasRowError(flags)
    val hasColumnError get() = ValidationError.hasColumnError(flags)
    val hasCageError get() = ValidationError.hasCageError(flags)
}

/**
 * Result of grid validation.
 */
data class ValidationResult(
    val errors: List<CellError>,
    val errorCount: Int
) {
    val isValid get() = errorCount == 0
}
