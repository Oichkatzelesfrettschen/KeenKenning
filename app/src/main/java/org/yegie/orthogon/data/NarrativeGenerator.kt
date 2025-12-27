/*
 * NarrativeGenerator.kt: AI-powered narrative generation for Story mode
 *
 * Hybrid approach:
 * - Primary: CharLSTM ONNX model (892KB) trained on text adventure + D&D corpus
 * - Fallback: Template-based generation using phrase database (5KB)
 *
 * The CharLSTM generates fantasy/RPG flavored narratives with text adventure style.
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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.nio.LongBuffer
import kotlin.math.exp
import kotlin.random.Random

/**
 * CharLSTM-based narrative generator for Story mode.
 *
 * Generates text adventure style narratives based on:
 * - Chapter theme (exploration, puzzle mastery, etc.)
 * - Player performance
 * - Story progression
 *
 * Uses character-level LSTM trained on Zork, D&D, and fantasy corpora.
 */
class NarrativeGenerator private constructor(private val context: Context) {

    companion object {
        private const val TAG = "NarrativeGen"
        private const val MODEL_FILENAME = "narrative_model.onnx"
        private const val VOCAB_FILENAME = "narrative_vocab.json"
        private const val PHRASE_DB_FILENAME = "phrase_database.json"
        private const val SEQ_LENGTH = 64
        private const val MAX_NEW_CHARS = 80
        private const val TEMPERATURE = 0.8f

        @Volatile
        @android.annotation.SuppressLint("StaticFieldLeak")  // Safe: stores applicationContext, not Activity
        private var instance: NarrativeGenerator? = null

        fun getInstance(context: Context): NarrativeGenerator {
            return instance ?: synchronized(this) {
                instance ?: NarrativeGenerator(context.applicationContext).also {
                    instance = it
                }
            }
        }
    }

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var charToIdx: Map<String, Int> = emptyMap()
    private var idxToChar: Map<Int, String> = emptyMap()
    private var phraseDb: PhraseDatabase? = null
    private var isModelLoaded = false

    init {
        loadPhraseDatabase()
        loadModel()
    }

    /**
     * Load phrase database for template-based fallback generation.
     */
    private fun loadPhraseDatabase() {
        try {
            val jsonText = context.assets.open(PHRASE_DB_FILENAME).bufferedReader().readText()
            val json = JSONObject(jsonText)
            phraseDb = PhraseDatabase(
                actionVerbs = json.getJSONArray("action_verbs").toStringList(),
                adjectives = json.getJSONArray("adjectives").toStringList(),
                nouns = json.getJSONArray("nouns").toStringList(),
                locations = json.getJSONArray("locations").toStringList(),
                characters = json.getJSONArray("characters").toStringList(),
                temporal = json.getJSONArray("temporal").toStringList(),
                victory = json.getJSONArray("victory").toStringList(),
                introPatterns = json.getJSONArray("intro_patterns").toStringList(),
                outroPatterns = json.getJSONArray("outro_patterns").toStringList()
            )
            Log.i(TAG, "Phrase database loaded: ${phraseDb?.adjectives?.size} adjectives")
        } catch (e: Exception) {
            Log.w(TAG, "Could not load phrase database", e)
        }
    }

    /**
     * Load the CharLSTM model and character vocabulary.
     */
    private fun loadModel() {
        try {
            // Copy model to cache if needed (for external data files)
            val cacheDir = context.cacheDir
            val modelFile = File(cacheDir, MODEL_FILENAME)

            if (!modelFile.exists()) {
                context.assets.open(MODEL_FILENAME).use { input ->
                    modelFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            }

            ortEnv = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions()
            opts.setIntraOpNumThreads(2) // Limit threads for mobile
            ortSession = ortEnv?.createSession(modelFile.absolutePath, opts)

            loadVocabulary()
            isModelLoaded = true
            Log.i(TAG, "CharLSTM model loaded successfully")
        } catch (e: Exception) {
            Log.w(TAG, "Could not load CharLSTM model, using template fallback", e)
            isModelLoaded = false
        }
    }

    /**
     * Load character-level vocabulary for model encoding.
     */
    private fun loadVocabulary() {
        try {
            val jsonText = context.assets.open(VOCAB_FILENAME).bufferedReader().readText()
            val json = JSONObject(jsonText)
            val char2idx = mutableMapOf<String, Int>()
            val idx2char = mutableMapOf<Int, String>()

            json.keys().forEach { key ->
                val idx = json.getInt(key)
                char2idx[key] = idx
                idx2char[idx] = key
            }

            charToIdx = char2idx
            idxToChar = idx2char
            Log.i(TAG, "Character vocabulary loaded: ${charToIdx.size} tokens")
        } catch (e: Exception) {
            Log.w(TAG, "Could not load vocabulary", e)
        }
    }

    /**
     * Generate a story intro for a chapter.
     *
     * Uses phrase database template generation for reliable narratives.
     * CharLSTM model is loaded but template approach is used for v1.0
     * to ensure consistent quality.
     *
     * @param chapterTitle The chapter title
     * @param theme The chapter theme icon name
     * @param difficulty The puzzle difficulty (0-4)
     * @return Generated narrative text
     */
    suspend fun generateIntro(
        chapterTitle: String,
        theme: String,
        difficulty: Int
    ): String = withContext(Dispatchers.Default) {
        // Use template-based generation from phrase database
        generateFromTemplate(isIntro = true, chapterTitle, theme, difficulty)
    }

    /**
     * Generate a story outro after completing a chapter.
     *
     * @param chapterTitle The chapter title
     * @param success Whether the chapter was completed successfully
     * @param timeSeconds Time taken to complete
     * @return Generated narrative text
     */
    suspend fun generateOutro(
        chapterTitle: String,
        success: Boolean,
        @Suppress("UNUSED_PARAMETER") timeSeconds: Int
    ): String = withContext(Dispatchers.Default) {
        generateFromTemplate(
            isIntro = false,
            title = chapterTitle,
            theme = if (success) "victory" else "struggle",
            difficulty = if (success) 0 else 4
        )
    }

    // =========================================================================
    // Template-based generation using phrase database
    // =========================================================================

    /**
     * Generate narrative from phrase database templates.
     *
     * Fills template placeholders with random selections from vocabulary:
     * {temporal}, {adj}, {location}, {noun}, {character}, {action}, {victory}
     */
    @Suppress("UNUSED_PARAMETER")
    private fun generateFromTemplate(
        isIntro: Boolean,
        title: String,
        theme: String,  // Reserved for theme-specific patterns
        difficulty: Int
    ): String {
        val db = phraseDb ?: return getDefaultNarrative(isIntro, title)

        // Select random pattern
        val patterns = if (isIntro) db.introPatterns else db.outroPatterns
        val pattern = patterns.randomOrNull() ?: return getDefaultNarrative(isIntro, title)

        // Difficulty word for substitution
        val difficultyWord = when (difficulty) {
            0 -> "simple"
            1 -> "moderate"
            2 -> "challenging"
            3 -> "formidable"
            else -> "legendary"
        }

        // Fill placeholders
        val filled = pattern
            .replace("{temporal}", db.temporal.random())
            .replace("{adj}", db.adjectives.random())
            .replace("{adj2}", db.adjectives.random())
            .replace("{location}", db.locations.random())
            .replace("{noun}", db.nouns.random())
            .replace("{character}", db.characters.random())
            .replace("{action}", db.actionVerbs.random())
            .replace("{victory}", db.victory.random())
            .replace("{difficulty}", difficultyWord)
            .replace("{size}", Random.nextInt(4, 10).toString())
            .replace("{time}", Random.nextInt(30, 300).toString())
            .replace("{reward}", "the Mark of the ${db.adjectives.random().replaceFirstChar { it.uppercase() }} ${db.characters.random().replaceFirstChar { it.uppercase() }}")

        return filled
    }

    /**
     * Default narrative when database is unavailable.
     */
    private fun getDefaultNarrative(isIntro: Boolean, title: String): String {
        return if (isIntro) {
            "The chapter \"$title\" awaits. Numbers dance in patterns before you."
        } else {
            "Victory! The chapter \"$title\" is complete."
        }
    }

    // =========================================================================
    // CharLSTM inference (reserved for future use)
    // =========================================================================

    /**
     * Run character-level autoregressive generation using CharLSTM.
     * Currently unused - phrase database provides better consistency.
     */
    @Suppress("unused")
    private fun generateWithCharLSTM(seed: String): String {
        val env = ortEnv ?: return seed
        val session = ortSession ?: return seed
        if (charToIdx.isEmpty()) return seed

        try {
            // Encode seed to character indices
            val seedIds = seed.map { c ->
                (charToIdx[c.toString()] ?: charToIdx["<unk>"] ?: 3).toLong()
            }.toLongArray()

            // Pad/truncate to SEQ_LENGTH
            val inputIds = if (seedIds.size >= SEQ_LENGTH) {
                seedIds.takeLast(SEQ_LENGTH).toLongArray()
            } else {
                LongArray(SEQ_LENGTH) { i ->
                    if (i < SEQ_LENGTH - seedIds.size) 0L else seedIds[i - (SEQ_LENGTH - seedIds.size)]
                }
            }

            val generated = StringBuilder(seed)

            // Generate characters
            for (step in 0 until MAX_NEW_CHARS) {
                val inputBuffer = LongBuffer.wrap(inputIds)
                val inputTensor = OnnxTensor.createTensor(
                    env, inputBuffer, longArrayOf(1, SEQ_LENGTH.toLong())
                )

                val results = session.run(mapOf("input_ids" to inputTensor))
                val logitsTensor = results[0] as OnnxTensor
                val logits = logitsTensor.floatBuffer

                // Sample next character
                val nextCharIdx = sampleCharacter(logits, TEMPERATURE)

                inputTensor.close()
                results.close()

                // Check for EOS
                if (nextCharIdx == charToIdx["<eos>"]) break

                val nextChar = idxToChar[nextCharIdx]
                if (nextChar != null && !nextChar.startsWith("<")) {
                    generated.append(nextChar)

                    // Shift input window
                    for (i in 0 until SEQ_LENGTH - 1) {
                        inputIds[i] = inputIds[i + 1]
                    }
                    inputIds[SEQ_LENGTH - 1] = nextCharIdx.toLong()
                }
            }

            return generated.toString()
        } catch (e: Exception) {
            Log.e(TAG, "CharLSTM generation failed", e)
            return seed
        }
    }

    /**
     * Sample next character from logits with temperature.
     */
    private fun sampleCharacter(logits: java.nio.FloatBuffer, temperature: Float): Int {
        val vocabSize = charToIdx.size
        // Get logits for last position (seq_len - 1)
        val startIdx = (SEQ_LENGTH - 1) * vocabSize

        val probs = FloatArray(vocabSize)
        var maxLogit = Float.NEGATIVE_INFINITY

        for (i in 0 until vocabSize) {
            val idx = startIdx + i
            if (idx < logits.capacity()) {
                probs[i] = logits.get(idx) / temperature
                if (probs[i] > maxLogit) maxLogit = probs[i]
            }
        }

        // Softmax
        var sumExp = 0f
        for (i in probs.indices) {
            probs[i] = exp(probs[i] - maxLogit)
            sumExp += probs[i]
        }
        for (i in probs.indices) {
            probs[i] /= sumExp
        }

        // Sample from distribution
        val r = Random.nextFloat()
        var cumulative = 0f
        for (i in probs.indices) {
            cumulative += probs[i]
            if (r <= cumulative) return i
        }
        return vocabSize - 1
    }

    /**
     * Close the session and release resources.
     */
    fun close() {
        ortSession?.close()
        ortSession = null
        ortEnv?.close()
        ortEnv = null
        phraseDb = null
        isModelLoaded = false
        instance = null
    }
}

/**
 * Container for phrase database loaded from JSON assets.
 */
private data class PhraseDatabase(
    val actionVerbs: List<String>,
    val adjectives: List<String>,
    val nouns: List<String>,
    val locations: List<String>,
    val characters: List<String>,
    val temporal: List<String>,
    val victory: List<String>,
    val introPatterns: List<String>,
    val outroPatterns: List<String>
)

/**
 * Extension to convert JSONArray to List<String>.
 */
private fun org.json.JSONArray.toStringList(): List<String> {
    val list = mutableListOf<String>()
    for (i in 0 until length()) {
        list.add(getString(i))
    }
    return list
}
