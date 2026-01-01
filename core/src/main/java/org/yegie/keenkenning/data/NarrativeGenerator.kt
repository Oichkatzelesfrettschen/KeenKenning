/*
 * NarrativeGenerator.kt: Story narrative generation interface
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning.data

interface NarrativeGenerator {
    suspend fun generateIntro(
        chapterTitle: String,
        theme: String,
        difficulty: Int
    ): String

    suspend fun generateOutro(
        chapterTitle: String,
        success: Boolean,
        timeSeconds: Int
    ): String

    fun close()
}
