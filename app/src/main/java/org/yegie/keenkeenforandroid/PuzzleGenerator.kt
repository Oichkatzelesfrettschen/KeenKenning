/*
 * PuzzleGenerator.kt: Kotlin facade for puzzle generation with type-safe results
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKeen Contributors
 *
 * Wraps the legacy KeenModelBuilder with sealed result types and
 * structured error handling. This is the recommended entry point
 * for new code requiring puzzle generation.
 */

package org.yegie.keenkeenforandroid

import android.content.Context
import android.util.Log
import org.yegie.keenkeenforandroid.data.GameMode
import org.yegie.keenkeenforandroid.data.PuzzleGenerationResult

/**
 * Configuration for puzzle generation.
 */
data class GenerationConfig(
    val size: Int,
    val difficulty: Int,
    val multiplicationOnly: Int = 0,
    val seed: Long,
    val useAI: Boolean = false,
    val modeFlags: Int = 0
) {
    init {
        require(size in 3..16) { "Size must be between 3 and 16, got $size" }
        require(difficulty in 0..4) { "Difficulty must be between 0 and 4, got $difficulty" }
    }

    companion object {
        fun fromGameMode(
            size: Int,
            difficulty: Int,
            seed: Long,
            gameMode: GameMode,
            useAI: Boolean = false
        ): GenerationConfig {
            return GenerationConfig(
                size = size,
                difficulty = difficulty,
                multiplicationOnly = if (gameMode == GameMode.MULTIPLICATION_ONLY) 1 else 0,
                seed = seed,
                useAI = useAI,
                modeFlags = gameMode.cFlags
            )
        }
    }
}

/**
 * Type-safe puzzle generator using sealed results.
 *
 * Usage:
 * ```kotlin
 * val generator = PuzzleGenerator()
 * when (val result = generator.generate(context, config)) {
 *     is PuzzleGenerationResult.Success -> {
 *         val model = result.model
 *         // use model
 *     }
 *     is PuzzleGenerationResult.Failure -> {
 *         Log.e("GEN", "Failed: ${result.message}")
 *     }
 * }
 * ```
 */
class PuzzleGenerator {

    private val legacyBuilder = KeenModelBuilder()

    /**
     * Generate a puzzle with type-safe result handling.
     *
     * @param context Android context for AI model loading
     * @param config Generation configuration
     * @return PuzzleGenerationResult.Success or specific Failure variant
     */
    fun generate(context: Context, config: GenerationConfig): PuzzleGenerationResult {
        val startTime = System.currentTimeMillis()

        // Validate parameters
        if (config.size !in 3..16) {
            return PuzzleGenerationResult.Failure.InvalidParameters(
                message = "Grid size must be between 3 and 16",
                paramName = "size",
                providedValue = config.size
            )
        }

        return try {
            val model = legacyBuilder.build(
                context,
                config.size,
                config.difficulty,
                config.multiplicationOnly,
                config.seed,
                config.useAI,
                config.modeFlags
            )

            if (model != null) {
                val elapsed = System.currentTimeMillis() - startTime
                Log.d("PuzzleGenerator", "Generated ${config.size}x${config.size} puzzle in ${elapsed}ms")

                PuzzleGenerationResult.Success(
                    model = model,
                    wasAiGenerated = model.wasAiGenerated(),
                    generationTimeMs = elapsed
                )
            } else {
                PuzzleGenerationResult.Failure.NativeGenerationFailed(
                    message = "KeenModelBuilder returned null"
                )
            }
        } catch (e: NumberFormatException) {
            Log.e("PuzzleGenerator", "Parsing error: ${e.message}")
            PuzzleGenerationResult.Failure.ParsingFailed(
                message = "Failed to parse JNI payload: ${e.message}",
                rawPayload = null // Don't expose raw data in errors
            )
        } catch (e: StringIndexOutOfBoundsException) {
            Log.e("PuzzleGenerator", "Payload truncation: ${e.message}")
            PuzzleGenerationResult.Failure.ParsingFailed(
                message = "JNI payload was truncated or malformed",
                rawPayload = null
            )
        } catch (e: Exception) {
            Log.e("PuzzleGenerator", "Unexpected error: ${e.message}", e)
            PuzzleGenerationResult.Failure.NativeGenerationFailed(
                message = "Unexpected error during generation: ${e.message}"
            )
        }
    }

    /**
     * Generate with AI, falling back to native on failure.
     * Returns detailed information about which path was used.
     */
    fun generateWithAIFallback(
        context: Context,
        config: GenerationConfig
    ): PuzzleGenerationResult {
        // First try with AI if requested
        if (config.useAI) {
            val aiConfig = config.copy(useAI = true)
            val result = generate(context, aiConfig)

            if (result is PuzzleGenerationResult.Success) {
                return result
            }

            // Log AI failure and fallback
            Log.w("PuzzleGenerator", "AI generation failed, falling back to native")
        }

        // Fallback to native generation
        val nativeConfig = config.copy(useAI = false)
        return generate(context, nativeConfig)
    }

    companion object {
        @Volatile
        private var instance: PuzzleGenerator? = null

        /**
         * Get singleton instance for efficient reuse.
         */
        fun getInstance(): PuzzleGenerator {
            return instance ?: synchronized(this) {
                instance ?: PuzzleGenerator().also { instance = it }
            }
        }
    }
}

/**
 * Extension function for convenient generation from GameMode.
 */
fun PuzzleGenerator.generate(
    context: Context,
    size: Int,
    difficulty: Int,
    seed: Long,
    gameMode: GameMode,
    useAI: Boolean = false
): PuzzleGenerationResult {
    val config = GenerationConfig.fromGameMode(size, difficulty, seed, gameMode, useAI)
    return generate(context, config)
}
