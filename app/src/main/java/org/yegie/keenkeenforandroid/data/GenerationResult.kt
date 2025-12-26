/*
 * GenerationResult.kt: Sealed result types for puzzle generation
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKeen Contributors
 *
 * Provides type-safe error handling for JNI boundary and generation pipeline.
 * Replaces implicit null/empty string error signaling with explicit variants.
 */

package org.yegie.keenkeenforandroid.data

import org.yegie.keenkeenforandroid.KeenModel

/**
 * Sealed hierarchy for puzzle generation results.
 * Eliminates ambiguous null/empty returns from JNI layer.
 */
sealed interface PuzzleGenerationResult {
    /**
     * Successful generation with a valid puzzle model.
     * @param model The generated puzzle
     * @param wasAiGenerated Whether AI was used for Latin square generation
     * @param generationTimeMs Time taken to generate (for profiling)
     */
    data class Success(
        val model: KeenModel,
        val wasAiGenerated: Boolean,
        val generationTimeMs: Long = 0
    ) : PuzzleGenerationResult

    /**
     * Generation failed with a specific reason.
     */
    sealed interface Failure : PuzzleGenerationResult {
        val message: String

        /** JNI layer returned null (grid invalid or solver failed) */
        data class NativeGenerationFailed(
            override val message: String = "Native generation returned null"
        ) : Failure

        /** AI model failed to produce valid Latin square */
        data class AiGenerationFailed(
            override val message: String = "AI generation failed",
            val fallbackAttempted: Boolean = false
        ) : Failure

        /** Parsing the JNI string payload failed */
        data class ParsingFailed(
            override val message: String,
            val rawPayload: String? = null
        ) : Failure

        /** Invalid parameters (size out of range, etc.) */
        data class InvalidParameters(
            override val message: String,
            val paramName: String,
            val providedValue: Any?
        ) : Failure

        /** ONNX model loading/initialization failed */
        data class ModelLoadFailed(
            override val message: String = "ONNX model failed to load",
            val cause: Throwable? = null
        ) : Failure
    }
}

/**
 * Sealed hierarchy for Latin square provider results.
 * Used by LatinSquareProvider implementations.
 */
sealed interface LatinSquareResult {
    /**
     * Valid Latin square grid.
     * @param grid Flat array of digits [size*size]
     * @param probabilities Optional probability tensor [batch][class][y][x]
     */
    data class Success(
        val grid: IntArray,
        val probabilities: Array<Array<Array<FloatArray>>>? = null
    ) : LatinSquareResult {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is Success) return false
            return grid.contentEquals(other.grid)
        }
        override fun hashCode(): Int = grid.contentHashCode()
    }

    sealed interface Failure : LatinSquareResult {
        val message: String

        data class InvalidGrid(
            override val message: String = "Generated grid is not a valid Latin square"
        ) : Failure

        data class GenerationTimeout(
            override val message: String = "Generation timed out"
        ) : Failure

        data class ModelError(
            override val message: String,
            val cause: Throwable? = null
        ) : Failure
    }
}

/**
 * Extension to check if result is successful.
 */
fun PuzzleGenerationResult.isSuccess(): Boolean = this is PuzzleGenerationResult.Success

/**
 * Extension to get model or null.
 */
fun PuzzleGenerationResult.getModelOrNull(): KeenModel? =
    (this as? PuzzleGenerationResult.Success)?.model

/**
 * Extension to get error message or null.
 */
fun PuzzleGenerationResult.getErrorOrNull(): String? =
    (this as? PuzzleGenerationResult.Failure)?.message
