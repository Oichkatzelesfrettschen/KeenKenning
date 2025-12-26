/*
 * JniResultParserTest.kt: Unit tests for JNI response parsing
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKeen Contributors
 */

package org.yegie.keenkeenforandroid.data

import org.junit.Assert.*
import org.junit.Test

class JniResultParserTest {

    @Test
    fun `parse null response returns NullResponse`() {
        val result = JniResultParser.parse(null)
        assertTrue(result is JniResult.NullResponse)
    }

    @Test
    fun `parse OK prefix extracts payload`() {
        val result = JniResultParser.parse("OK:some_payload_data")
        assertTrue(result is JniResult.Success)
        assertEquals("some_payload_data", (result as JniResult.Success).payload)
    }

    @Test
    fun `parse OK prefix with complex payload`() {
        val payload = "00,00,01,01;a00006,m00012,123456789"
        val result = JniResultParser.parse("OK:$payload")
        assertTrue(result is JniResult.Success)
        assertEquals(payload, (result as JniResult.Success).payload)
    }

    @Test
    fun `parse ERR prefix extracts error code and message`() {
        val result = JniResultParser.parse("ERR:1:Grid array length does not match size*size")
        assertTrue(result is JniResult.Error)

        val error = result as JniResult.Error
        assertEquals(JniErrorCode.GRID_SIZE, error.errorCode)
        assertEquals("Grid array length does not match size*size", error.message)
    }

    @Test
    fun `parse ERR with invalid params code`() {
        val result = JniResultParser.parse("ERR:5:Size must be 3-16")
        assertTrue(result is JniResult.Error)

        val error = result as JniResult.Error
        assertEquals(JniErrorCode.INVALID_PARAMS, error.errorCode)
        assertEquals("Size must be 3-16", error.message)
    }

    @Test
    fun `parse ERR with memory error code`() {
        val result = JniResultParser.parse("ERR:4:Failed to allocate input grid")
        assertTrue(result is JniResult.Error)

        val error = result as JniResult.Error
        assertEquals(JniErrorCode.MEMORY, error.errorCode)
    }

    @Test
    fun `parse ERR with unknown code defaults to GENERATION_FAIL`() {
        val result = JniResultParser.parse("ERR:999:Unknown error type")
        assertTrue(result is JniResult.Error)

        val error = result as JniResult.Error
        assertEquals(JniErrorCode.GENERATION_FAIL, error.errorCode)
    }

    @Test
    fun `parse ERR with colons in message preserves full message`() {
        val result = JniResultParser.parse("ERR:3:Error: value was: 42")
        assertTrue(result is JniResult.Error)

        val error = result as JniResult.Error
        assertEquals(JniErrorCode.GENERATION_FAIL, error.errorCode)
        assertEquals("Error: value was: 42", error.message)
    }

    @Test
    fun `parse legacy format without prefix as success`() {
        val payload = "00,00,01,01;a00006,123456"
        val result = JniResultParser.parse(payload)
        assertTrue(result is JniResult.Success)
        assertEquals(payload, (result as JniResult.Success).payload)
    }

    @Test
    fun `parse empty string as success with empty payload`() {
        val result = JniResultParser.parse("")
        assertTrue(result is JniResult.Success)
        assertEquals("", (result as JniResult.Success).payload)
    }

    @Test
    fun `parse ERR without message returns malformed`() {
        val result = JniResultParser.parse("ERR:1")
        assertTrue(result is JniResult.MalformedResponse)
    }

    @Test
    fun `parse ERR with non-numeric code returns malformed`() {
        val result = JniResultParser.parse("ERR:abc:Some message")
        assertTrue(result is JniResult.MalformedResponse)
    }

    // Test conversion to PuzzleGenerationResult.Failure

    @Test
    fun `toGenerationFailure maps INVALID_PARAMS correctly`() {
        val jniResult = JniResult.Error(JniErrorCode.INVALID_PARAMS, "Size must be 3-16")
        val failure = JniResultParser.toGenerationFailure(jniResult)

        assertTrue(failure is PuzzleGenerationResult.Failure.InvalidParameters)
        assertEquals("Size must be 3-16", failure.message)
    }

    @Test
    fun `toGenerationFailure maps INVALID_GRID to AiGenerationFailed`() {
        val jniResult = JniResult.Error(JniErrorCode.INVALID_GRID, "AI grid rejected")
        val failure = JniResultParser.toGenerationFailure(jniResult)

        assertTrue(failure is PuzzleGenerationResult.Failure.AiGenerationFailed)
        val aiFailure = failure as PuzzleGenerationResult.Failure.AiGenerationFailed
        assertFalse(aiFailure.fallbackAttempted)
    }

    @Test
    fun `toGenerationFailure maps MEMORY to NativeGenerationFailed`() {
        val jniResult = JniResult.Error(JniErrorCode.MEMORY, "Allocation failed")
        val failure = JniResultParser.toGenerationFailure(jniResult)

        assertTrue(failure is PuzzleGenerationResult.Failure.NativeGenerationFailed)
        assertTrue(failure.message.contains("Memory"))
    }

    @Test
    fun `toGenerationFailure handles NullResponse`() {
        val failure = JniResultParser.toGenerationFailure(JniResult.NullResponse)

        assertTrue(failure is PuzzleGenerationResult.Failure.NativeGenerationFailed)
        assertTrue(failure.message.contains("null"))
    }

    @Test
    fun `toGenerationFailure handles MalformedResponse`() {
        val jniResult = JniResult.MalformedResponse("ERR:bad")
        val failure = JniResultParser.toGenerationFailure(jniResult)

        assertTrue(failure is PuzzleGenerationResult.Failure.ParsingFailed)
        val parsingFailure = failure as PuzzleGenerationResult.Failure.ParsingFailed
        assertEquals("ERR:bad", parsingFailure.rawPayload)
    }

    // Test JniErrorCode.fromCode

    @Test
    fun `JniErrorCode fromCode returns correct enum`() {
        assertEquals(JniErrorCode.NONE, JniErrorCode.fromCode(0))
        assertEquals(JniErrorCode.GRID_SIZE, JniErrorCode.fromCode(1))
        assertEquals(JniErrorCode.INVALID_GRID, JniErrorCode.fromCode(2))
        assertEquals(JniErrorCode.GENERATION_FAIL, JniErrorCode.fromCode(3))
        assertEquals(JniErrorCode.MEMORY, JniErrorCode.fromCode(4))
        assertEquals(JniErrorCode.INVALID_PARAMS, JniErrorCode.fromCode(5))
        assertEquals(JniErrorCode.CLUE_GENERATION, JniErrorCode.fromCode(6))
    }

    @Test
    fun `JniErrorCode fromCode defaults unknown to GENERATION_FAIL`() {
        assertEquals(JniErrorCode.GENERATION_FAIL, JniErrorCode.fromCode(100))
        assertEquals(JniErrorCode.GENERATION_FAIL, JniErrorCode.fromCode(-1))
    }
}
