/*
 * StoryActivity.kt: Stub for Classik flavor (no Story Mode)
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 *
 * Classik flavor: Story Mode is not available.
 * This stub exists for compile compatibility - it should never be launched
 * since GameMode.availableModes() excludes STORY in Classik.
 */

package org.yegie.keenkenning

import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity

/**
 * Stub StoryActivity for Classik flavor.
 * Story Mode is not available in Keen Classik.
 */
class StoryActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Story Mode is not available in Classik - close immediately
        Toast.makeText(this, "Story Mode requires Keen Kenning", Toast.LENGTH_SHORT).show()
        finish()
    }
}
