/*
 * PerfTrace.kt: Thin wrapper around androidx.tracing for structured sections
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning.perf

import androidx.tracing.Trace

object PerfTrace {
    inline fun <T> section(name: String, block: () -> T): T {
        val enabled = runCatching { Trace.isEnabled() }.getOrDefault(false)
        if (!enabled) {
            return block()
        }
        runCatching { Trace.beginSection(name) }
        return try {
            block()
        } finally {
            runCatching { Trace.endSection() }
        }
    }
}
