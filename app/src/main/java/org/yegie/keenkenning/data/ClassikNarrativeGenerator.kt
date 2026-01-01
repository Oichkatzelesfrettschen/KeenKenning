/*
 * ClassikNarrativeGenerator.kt: Stub for Classik flavor (no Story Mode)
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
 * Stub narrative generator for Classik flavor.
 * Story Mode is not available in Keen Classik.
 */
@Suppress("UNUSED_PARAMETER")
class ClassikNarrativeGenerator private constructor(private val context: Context) : NarrativeGenerator {

    companion object {
        @Volatile
        @android.annotation.SuppressLint("StaticFieldLeak")
        private var instance: ClassikNarrativeGenerator? = null

        @JvmStatic
        fun getInstance(context: Context): ClassikNarrativeGenerator {
            return instance ?: synchronized(this) {
                instance ?: ClassikNarrativeGenerator(context.applicationContext).also {
                    instance = it
                }
            }
        }
    }

    override suspend fun generateIntro(
        chapterTitle: String,
        theme: String,
        difficulty: Int
    ): String = "Story Mode requires Keen Kenning"

    override suspend fun generateOutro(
        chapterTitle: String,
        success: Boolean,
        timeSeconds: Int
    ): String = "Story Mode requires Keen Kenning"

    override fun close() {
        instance = null
    }
}
