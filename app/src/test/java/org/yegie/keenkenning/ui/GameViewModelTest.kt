package org.yegie.keenkenning.ui

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.Timeout
import org.yegie.keenkenning.KeenModel
import org.yegie.keenkenning.KeenModel.GridCell
import org.yegie.keenkenning.KeenModel.Zone
import java.util.concurrent.TimeUnit

@OptIn(ExperimentalCoroutinesApi::class)
class GameViewModelTest {

    @get:Rule
    val timeout: Timeout = Timeout.seconds(5)

    private val dispatcher = StandardTestDispatcher()

    @Before
    fun setup() {
        Dispatchers.setMain(dispatcher)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `loadModel updates uiState and restores input mode`() = runTest {
        val viewModel = GameViewModel()
        
        try {
            // Mock data
            val size = 3
            val zones = arrayOf(Zone(Zone.Type.ADD, 5, 0))
            val grid = Array(size) { Array(size) { GridCell(1, zones[0]) } }
            val model = KeenModel(size, zones, grid, false)
            
            // Simulate saved state: finalGuess = false means Note Mode is active
            model.toggleFinalGuess() // Set to false (Note Mode)
            assertFalse(model.finalGuess)

            viewModel.loadModel(model)

            assertEquals(size, viewModel.uiState.value.size)
            // Verify state restoration logic
            assertTrue("Input mode should be restored from model", viewModel.uiState.value.isInputtingNotes)
        } finally {
            // CRITICAL: Stop the infinite timer loop to allow runTest to complete
            viewModel.pauseTimer()
        }
    }
}
