/*
 * JniResultParser.kt: Parser for structured JNI response envelopes
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKeen Contributors
 *
 * Parses the structured response format from native JNI functions:
 *   Success: "OK:payload_data"
 *   Error:   "ERR:code:message"
 *
 * Maps native error codes to the PuzzleGenerationResult.Failure hierarchy
 * for type-safe error handling in Kotlin.
 */

package org.yegie.keenkeenforandroid.data

/**
 * Error codes matching jni_error_codes.h definitions.
 * Must be kept in sync with native header.
 */
enum class JniErrorCode(val code: Int) {
    NONE(0),
    GRID_SIZE(1),
    INVALID_GRID(2),
    GENERATION_FAIL(3),
    MEMORY(4),
    INVALID_PARAMS(5),
    CLUE_GENERATION(6);

    companion object {
        fun fromCode(code: Int): JniErrorCode =
            entries.find { it.code == code } ?: GENERATION_FAIL
    }
}

/**
 * Parsed result from JNI layer.
 */
sealed interface JniResult {
    /**
     * Successful JNI call with payload data.
     */
    data class Success(val payload: String) : JniResult

    /**
     * JNI call returned an error.
     */
    data class Error(
        val errorCode: JniErrorCode,
        val message: String
    ) : JniResult

    /**
     * JNI returned null (legacy behavior or catastrophic failure).
     */
    data object NullResponse : JniResult

    /**
     * JNI response format was unrecognized.
     */
    data class MalformedResponse(val rawResponse: String) : JniResult
}

/**
 * Parser for structured JNI responses.
 */
object JniResultParser {
    private const val PREFIX_OK = "OK:"
    private const val PREFIX_ERR = "ERR:"

    /**
     * Parse a JNI response string into a structured result.
     *
     * @param response Raw string from JNI, or null
     * @return Parsed JniResult
     */
    fun parse(response: String?): JniResult {
        if (response == null) {
            return JniResult.NullResponse
        }

        return when {
            response.startsWith(PREFIX_OK) -> {
                val payload = response.substring(PREFIX_OK.length)
                JniResult.Success(payload)
            }
            response.startsWith(PREFIX_ERR) -> {
                parseError(response)
            }
            else -> {
                // Legacy format (no prefix) - treat as success for backward compatibility
                // This handles responses from older native code that hasn't been updated
                JniResult.Success(response)
            }
        }
    }

    /**
     * Parse error response format: "ERR:code:message"
     */
    private fun parseError(response: String): JniResult {
        val parts = response.substring(PREFIX_ERR.length).split(":", limit = 2)

        if (parts.size < 2) {
            return JniResult.MalformedResponse(response)
        }

        val code = parts[0].toIntOrNull() ?: return JniResult.MalformedResponse(response)
        val message = parts[1]

        return JniResult.Error(
            errorCode = JniErrorCode.fromCode(code),
            message = message
        )
    }

    /**
     * Convert JniResult to PuzzleGenerationResult.Failure.
     * Used when JNI returns an error that needs to be surfaced to the UI.
     */
    fun toGenerationFailure(result: JniResult): PuzzleGenerationResult.Failure {
        return when (result) {
            is JniResult.Error -> {
                when (result.errorCode) {
                    JniErrorCode.INVALID_PARAMS -> PuzzleGenerationResult.Failure.InvalidParameters(
                        message = result.message,
                        paramName = "native",
                        providedValue = null
                    )
                    JniErrorCode.GRID_SIZE -> PuzzleGenerationResult.Failure.InvalidParameters(
                        message = result.message,
                        paramName = "gridSize",
                        providedValue = null
                    )
                    JniErrorCode.INVALID_GRID -> PuzzleGenerationResult.Failure.AiGenerationFailed(
                        message = result.message,
                        fallbackAttempted = false
                    )
                    JniErrorCode.MEMORY -> PuzzleGenerationResult.Failure.NativeGenerationFailed(
                        message = "Memory allocation failed: ${result.message}"
                    )
                    JniErrorCode.GENERATION_FAIL,
                    JniErrorCode.CLUE_GENERATION -> PuzzleGenerationResult.Failure.NativeGenerationFailed(
                        message = result.message
                    )
                    JniErrorCode.NONE -> PuzzleGenerationResult.Failure.NativeGenerationFailed(
                        message = "Unknown error: ${result.message}"
                    )
                }
            }
            is JniResult.NullResponse -> PuzzleGenerationResult.Failure.NativeGenerationFailed(
                message = "JNI returned null (possible native crash)"
            )
            is JniResult.MalformedResponse -> PuzzleGenerationResult.Failure.ParsingFailed(
                message = "Malformed JNI response format",
                rawPayload = result.rawResponse
            )
            is JniResult.Success -> {
                // This shouldn't happen - caller should check for success first
                PuzzleGenerationResult.Failure.NativeGenerationFailed(
                    message = "Internal error: Success treated as failure"
                )
            }
        }
    }
}
