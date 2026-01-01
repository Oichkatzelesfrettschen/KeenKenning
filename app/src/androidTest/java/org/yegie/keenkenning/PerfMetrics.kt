package org.yegie.keenkenning

import android.content.Context
import android.os.Debug
import androidx.test.platform.app.InstrumentationRegistry
import java.io.FileInputStream

data class MemorySnapshot(
    val pssKb: Long,
    val nativeHeapKb: Long,
    val javaHeapKb: Long
)

object PerfMetrics {
    fun captureMemorySnapshot(): MemorySnapshot {
        val runtime = Runtime.getRuntime()
        val javaHeapBytes = runtime.totalMemory() - runtime.freeMemory()
        val nativeHeapBytes = Debug.getNativeHeapAllocatedSize()

        return MemorySnapshot(
            pssKb = Debug.getPss(),
            nativeHeapKb = nativeHeapBytes / 1024,
            javaHeapKb = javaHeapBytes / 1024
        )
    }

    fun dumpGfxInfo(context: Context): String {
        val uiAutomation = InstrumentationRegistry.getInstrumentation().uiAutomation
        val pfd = uiAutomation.executeShellCommand("dumpsys gfxinfo ${context.packageName}")
        pfd.use { descriptor ->
            FileInputStream(descriptor.fileDescriptor).bufferedReader().use { reader ->
                return reader.readText()
            }
        }
    }
}
