/*
 * SkillModelInference.kt: ONNX-based skill prediction for adaptive difficulty
 *
 * Uses a lightweight 3-layer MLP (~50KB) to predict player skill from
 * performance metrics. Falls back to heuristic calculation if model unavailable.
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 */

package org.yegie.orthogon.data

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Input features for skill prediction.
 * All values should be raw (unnormalized) - normalization happens internally.
 */
data class SkillFeatures(
    /** Actual solve time / expected time (e.g., 0.8 = faster, 1.5 = slower) */
    val solveTimeRatio: Float,
    /** Errors made / total cells in puzzle */
    val errorRate: Float,
    /** Hints used / puzzles played recently */
    val hintRate: Float,
    /** Undos used / total cells */
    val undoRate: Float,
    /** Current win streak */
    val streak: Int,
    /** Grid size of current/recent puzzle */
    val puzzleSize: Int,
    /** Difficulty level (0-4) */
    val difficulty: Int,
    /** Puzzles completed this session */
    val sessionPuzzles: Int
) {
    /**
     * Normalize features to 0-1 range for model input.
     */
    fun toNormalizedArray(): FloatArray {
        return floatArrayOf(
            normalize(solveTimeRatio, 0f, 3f),
            normalize(errorRate, 0f, 0.5f),
            normalize(hintRate, 0f, 1f),
            normalize(undoRate, 0f, 0.5f),
            normalize(streak.toFloat(), 0f, 20f),
            normalize(puzzleSize.toFloat(), 3f, 16f),
            normalize(difficulty.toFloat(), 0f, 4f),
            normalize(sessionPuzzles.toFloat(), 0f, 50f)
        )
    }

    private fun normalize(value: Float, minVal: Float, maxVal: Float): Float {
        val normalized = (value - minVal) / (maxVal - minVal + 1e-8f)
        return normalized.coerceIn(0f, 1f)
    }
}

/**
 * Skill model inference using ONNX Runtime.
 *
 * Thread-safe: can be called from any thread.
 */
class SkillModelInference(private val context: Context) {

    companion object {
        private const val TAG = "SkillModel"
        private const val MODEL_FILENAME = "skill_model.onnx"
        private const val NUM_FEATURES = 8

        @Volatile
        @android.annotation.SuppressLint("StaticFieldLeak")  // Safe: stores applicationContext, not Activity
        private var instance: SkillModelInference? = null

        fun getInstance(context: Context): SkillModelInference {
            return instance ?: synchronized(this) {
                instance ?: SkillModelInference(context.applicationContext).also {
                    instance = it
                }
            }
        }
    }

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var isModelLoaded = false

    init {
        loadModel()
    }

    /**
     * Load the ONNX model from assets.
     */
    private fun loadModel() {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val modelBytes = context.assets.open(MODEL_FILENAME).use { it.readBytes() }
            ortSession = ortEnv?.createSession(modelBytes)
            isModelLoaded = true
            Log.i(TAG, "Skill model loaded successfully")
        } catch (e: Exception) {
            Log.w(TAG, "Could not load skill model, using heuristic fallback", e)
            isModelLoaded = false
        }
    }

    /**
     * Predict skill score from features.
     *
     * @param features Player performance features
     * @return Skill score in range [0.0, 1.0]
     */
    fun predictSkill(features: SkillFeatures): Float {
        return if (isModelLoaded && ortSession != null) {
            predictWithModel(features)
        } else {
            predictHeuristic(features)
        }
    }

    /**
     * Run inference with the ONNX model.
     */
    private fun predictWithModel(features: SkillFeatures): Float {
        val inputArray = features.toNormalizedArray()
        val env = ortEnv ?: return predictHeuristic(features)
        val session = ortSession ?: return predictHeuristic(features)

        try {
            // Create input tensor: shape [1, 8]
            val inputBuffer = FloatBuffer.wrap(inputArray)
            val inputTensor = OnnxTensor.createTensor(
                env,
                inputBuffer,
                longArrayOf(1, NUM_FEATURES.toLong())
            )

            // Run inference
            val inputs = mapOf("features" to inputTensor)
            val results = session.run(inputs)

            // Extract output
            val outputTensor = results[0] as OnnxTensor
            val outputArray = outputTensor.floatBuffer.array()
            val skillScore = outputArray[0]

            // Clean up
            inputTensor.close()
            results.close()

            return skillScore.coerceIn(0f, 1f)
        } catch (e: Exception) {
            Log.e(TAG, "Model inference failed, using heuristic", e)
            return predictHeuristic(features)
        }
    }

    /**
     * Heuristic skill calculation when model is unavailable.
     * Mirrors the logic in UserStatsManager.computeSkillScore().
     */
    private fun predictHeuristic(features: SkillFeatures): Float {
        // Time factor (40%): faster = higher
        val timeFactor = (1f / max(features.solveTimeRatio, 0.1f)).coerceIn(0.2f, 2f)

        // Error penalty (15%)
        val errorPenalty = min(features.errorRate * 2f, 0.4f)

        // Hint penalty (15%)
        val hintPenalty = min(features.hintRate * 0.15f, 0.5f)

        // Undo penalty (10%)
        val undoPenalty = min(features.undoRate * 1.5f, 0.3f)

        // Difficulty bonus (20%)
        val difficultyBonus = features.difficulty * 0.2f

        // Streak bonus (soft cap at +0.15)
        val streakBonus = (1f - exp(-features.streak * 0.1f)) * 0.15f

        // Weighted combination
        val rawScore = (timeFactor * 0.4f) +
                       (difficultyBonus * 0.2f) +
                       (streakBonus) -
                       (hintPenalty * 0.15f) -
                       (errorPenalty * 0.15f) -
                       (undoPenalty * 0.1f)

        return rawScore.coerceIn(0f, 1f)
    }

    /**
     * Map skill score to recommended difficulty level.
     *
     * @param skillScore Skill in [0, 1]
     * @param gridSize Current grid size (larger = recommend easier)
     * @return Difficulty level 0-4
     */
    fun recommendDifficulty(skillScore: Float, gridSize: Int): Int {
        // Adjust for grid size (larger grids are inherently harder)
        val sizeAdjustment = when {
            gridSize >= 12 -> -0.1f
            gridSize <= 4 -> 0.1f
            else -> 0f
        }

        val adjusted = (skillScore + sizeAdjustment).coerceIn(0f, 1f)

        return when {
            adjusted < 0.25f -> 0  // Easy
            adjusted < 0.45f -> 1  // Normal
            adjusted < 0.65f -> 2  // Hard
            adjusted < 0.85f -> 3  // Insane
            else -> 4             // Ludicrous
        }
    }

    /**
     * Close the session and release resources.
     */
    fun close() {
        ortSession?.close()
        ortSession = null
        ortEnv?.close()
        ortEnv = null
        isModelLoaded = false
    }
}
