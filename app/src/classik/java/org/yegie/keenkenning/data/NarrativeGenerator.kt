/*
 * NarrativeGenerator.kt: Stub for Classik flavor (no Story Mode)
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 *
 * Classik flavor: Story Mode is not available.
 * This stub provides compile compatibility only.
 */

package org.yegie.keenkenning.data

import android.content.Context

/**
 * Stub NarrativeGenerator for Classik flavor.
 * Story Mode is not available in Keen Classik.
 */
@Suppress("UNUSED_PARAMETER")
class NarrativeGenerator private constructor(private val context: Context) {

    companion object {
        @Volatile
        @android.annotation.SuppressLint("StaticFieldLeak")
        private var instance: NarrativeGenerator? = null

        fun getInstance(context: Context): NarrativeGenerator {
            return instance ?: synchronized(this) {
                instance ?: NarrativeGenerator(context.applicationContext).also {
                    instance = it
                }
            }
        }
    }

    suspend fun generateIntro(
        chapterTitle: String,
        theme: String,
        difficulty: Int
    ): String = "Story Mode requires Keen Kenning"

    suspend fun generateOutro(
        chapterTitle: String,
        success: Boolean,
        timeSeconds: Int
    ): String = "Story Mode requires Keen Kenning"

    fun close() {
        instance = null
    }
}
