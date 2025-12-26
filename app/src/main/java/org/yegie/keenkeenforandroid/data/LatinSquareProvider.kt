/*
 * LatinSquareProvider.kt: Abstraction for Latin square generation strategies
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKeen Contributors
 *
 * Defines a unified interface for generating Latin squares, enabling
 * pluggable implementations: native backtracking, ONNX neural network,
 * or hybrid approaches.
 */

package org.yegie.keenkeenforandroid.data

import android.content.Context

/**
 * Strategy interface for Latin square generation.
 *
 * Implementations:
 * - NativeLatinSquareProvider: Uses C backtracking via JNI
 * - OnnxLatinSquareProvider: Uses ONNX Runtime neural network
 * - HybridLatinSquareProvider: Tries ONNX first, falls back to native
 */
interface LatinSquareProvider {
    /**
     * Generate a Latin square of the given size.
     *
     * @param size Grid dimension (3-16)
     * @param seed Random seed for deterministic generation (when applicable)
     * @return LatinSquareResult with grid or failure reason
     */
    suspend fun generate(size: Int, seed: Long): LatinSquareResult

    /**
     * Check if this provider supports the given grid size.
     * ONNX models may only support specific sizes they were trained on.
     */
    fun supportsSize(size: Int): Boolean

    /**
     * Provider name for logging/debugging.
     */
    val providerName: String

    /**
     * Whether this provider produces deterministic results for the same seed.
     * Neural network sampling may be non-deterministic.
     */
    val isDeterministic: Boolean
}

/**
 * Factory for creating Latin square providers.
 */
object LatinSquareProviderFactory {
    /**
     * Create the default provider chain: ONNX with native fallback.
     */
    fun createDefault(context: Context): LatinSquareProvider {
        return HybridLatinSquareProvider(
            primary = OnnxLatinSquareProvider(context),
            fallback = NativeLatinSquareProvider()
        )
    }

    /**
     * Create a native-only provider (fully deterministic).
     */
    fun createNative(): LatinSquareProvider = NativeLatinSquareProvider()

    /**
     * Create an ONNX-only provider (may fail for unsupported sizes).
     */
    fun createOnnx(context: Context): LatinSquareProvider = OnnxLatinSquareProvider(context)
}

/**
 * Native C backtracking implementation via JNI.
 * Fully deterministic given the same seed.
 */
class NativeLatinSquareProvider : LatinSquareProvider {
    override val providerName = "NativeBacktracking"
    override val isDeterministic = true

    override fun supportsSize(size: Int): Boolean = size in 3..16

    override suspend fun generate(size: Int, seed: Long): LatinSquareResult {
        // Native generation is handled in KeenModelBuilder.getLevelFromC
        // This provider is used when we want to separate Latin square generation
        // from clue generation. For now, return a placeholder indicating
        // native path should be used directly.
        return LatinSquareResult.Failure.GenerationTimeout(
            "NativeLatinSquareProvider requires direct JNI call path"
        )
    }
}

/**
 * ONNX neural network implementation.
 * Uses probability sampling which may be non-deterministic.
 */
class OnnxLatinSquareProvider(private val context: Context) : LatinSquareProvider {
    override val providerName = "OnnxNeuralNetwork"
    override val isDeterministic = false // Sampling introduces variance

    // ONNX model currently trained for sizes 3-9
    override fun supportsSize(size: Int): Boolean = size in 3..9

    override suspend fun generate(size: Int, seed: Long): LatinSquareResult {
        if (!supportsSize(size)) {
            return LatinSquareResult.Failure.InvalidGrid(
                "ONNX model does not support size $size"
            )
        }

        return try {
            val generator = org.yegie.keenkeenforandroid.NeuralKeenGenerator()
            val result = generator.generate(context, size)

            if (result?.grid != null) {
                LatinSquareResult.Success(
                    grid = result.grid,
                    probabilities = result.probabilities
                )
            } else {
                LatinSquareResult.Failure.ModelError("ONNX inference returned null")
            }
        } catch (e: Exception) {
            LatinSquareResult.Failure.ModelError(
                message = "ONNX generation failed: ${e.message}",
                cause = e
            )
        }
    }
}

/**
 * Hybrid provider that tries primary first, falls back on failure.
 */
class HybridLatinSquareProvider(
    private val primary: LatinSquareProvider,
    private val fallback: LatinSquareProvider
) : LatinSquareProvider {
    override val providerName = "Hybrid(${primary.providerName}->${fallback.providerName})"
    override val isDeterministic = primary.isDeterministic && fallback.isDeterministic

    override fun supportsSize(size: Int): Boolean =
        primary.supportsSize(size) || fallback.supportsSize(size)

    override suspend fun generate(size: Int, seed: Long): LatinSquareResult {
        // Try primary if it supports the size
        if (primary.supportsSize(size)) {
            val result = primary.generate(size, seed)
            if (result is LatinSquareResult.Success) {
                return result
            }
            // Log failure and try fallback
            android.util.Log.w("LatinSquare",
                "Primary provider failed: ${(result as? LatinSquareResult.Failure)?.message}")
        }

        // Fall back
        return if (fallback.supportsSize(size)) {
            fallback.generate(size, seed)
        } else {
            LatinSquareResult.Failure.InvalidGrid(
                "No provider supports size $size"
            )
        }
    }
}
