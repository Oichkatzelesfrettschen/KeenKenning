package org.yegie.keenkenning

import androidx.benchmark.junit4.BenchmarkRule
import androidx.benchmark.junit4.measureRepeated
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assume
import org.junit.Before
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.yegie.keenkenning.data.PuzzleGenerationResult
import org.yegie.keenkenning.data.GameMode
import org.yegie.keenkenning.data.JniResultParser
import org.yegie.keenkenning.data.ParseResult
import org.yegie.keenkenning.data.PuzzleParser
import org.yegie.keenkenning.ui.GameStateTransformer

@RunWith(AndroidJUnit4::class)
class KeenBenchmarkTest {
    @get:Rule
    val benchmarkRule = BenchmarkRule()

    private val samplePayload =
        "00,00,01,00,02,01,03,02,03;a00006,m00002,a00009,s00001,123456789"

    private val parsedPuzzle = (PuzzleParser.parse(samplePayload, 3) as? ParseResult.Success)
        ?.puzzle ?: error("Sample payload must parse for benchmarks")

    private val sampleModel = buildSampleModel()

    @Before
    fun assumeBenchmarkRunner() {
        val runnerName = InstrumentationRegistry.getInstrumentation()::class.java.name
        Assume.assumeTrue(
            "Benchmarks require AndroidBenchmarkRunner",
            runnerName == "androidx.benchmark.junit4.AndroidBenchmarkRunner"
        )
    }

    @Test
    fun benchmarkParsePayload() {
        benchmarkRule.measureRepeated {
            val result = PuzzleParser.parse(samplePayload, 3)
            if (result !is ParseResult.Success) {
                error("Parse failed during benchmark")
            }
        }
    }

    @Test
    fun benchmarkJniEnvelopeParse() {
        val wrapped = "OK:$samplePayload"
        benchmarkRule.measureRepeated {
            val result = JniResultParser.parse(wrapped)
            if (result is org.yegie.keenkenning.data.JniResult.Error) {
                error("JNI parse failed during benchmark")
            }
        }
    }

    @Test
    fun benchmarkRenderableTransform() {
        benchmarkRule.measureRepeated {
            GameStateTransformer.transform(
                model = sampleModel,
                gameMode = GameMode.STANDARD
            )
        }
    }

    @Test
    fun benchmarkLatinSquareValidation() {
        benchmarkRule.measureRepeated {
            PuzzleParser.isValidLatinSquare(parsedPuzzle)
        }
    }

    @Test
    fun benchmarkPuzzleGeneration() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val generator = PuzzleGenerator.getInstance()
        val config = GenerationConfig.fromGameMode(
            size = 3,
            difficulty = 1,
            seed = 42L,
            gameMode = GameMode.STANDARD,
            useAI = false
        )
        benchmarkRule.measureRepeated {
            val result = generator.generate(context, config)
            if (result !is PuzzleGenerationResult.Success) {
                error("Generation failed during benchmark")
            }
        }
    }

    @Test
    fun perfMetricsHooks() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val snapshot = PerfMetrics.captureMemorySnapshot()
        assertTrue(snapshot.pssKb >= 0)

        val gfxDump = PerfMetrics.dumpGfxInfo(context)
        assertTrue(gfxDump.isNotBlank())
    }

    private fun buildSampleModel(): KeenModel {
        val size = 3
        val zones = Array(size * size) { index ->
            KeenModel.Zone(KeenModel.Zone.Type.ADD, 1, index)
        }
        val grid = Array(size) { x ->
            Array(size) { y ->
                val index = x * size + y
                KeenModel.GridCell((index % size) + 1, zones[index])
            }
        }
        return KeenModel(size, zones, grid, false)
    }
}
