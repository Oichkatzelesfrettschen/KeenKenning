/*
 * StoryModeUnavailable.kt: Helper for unavailable Story Mode
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 *
 * Classik flavor: Story Mode is not available.
 * This helper centralizes the fallback behavior when Story Mode is unavailable.
 */

package org.yegie.keenkenning

import android.app.Activity
import android.widget.Toast

/**
 * Fallback handler for Story Mode in Classik.
 */
object StoryModeUnavailable {
    fun finish(activity: Activity) {
        Toast.makeText(activity, "Story Mode requires Keen Kenning", Toast.LENGTH_SHORT).show()
        activity.finish()
    }
}
