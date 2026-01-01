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
        return if (Trace.isEnabled()) {
            Trace.beginSection(name)
            try {
                block()
            } finally {
                Trace.endSection()
            }
        } else {
            block()
        }
    }
}
