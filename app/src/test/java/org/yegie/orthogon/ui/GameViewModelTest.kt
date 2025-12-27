package org.yegie.orthogon.ui

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.yegie.orthogon.KenKenModel
import org.yegie.orthogon.KenKenModel.GridCell
import org.yegie.orthogon.KenKenModel.Zone

@OptIn(ExperimentalCoroutinesApi::class)
class GameViewModelTest {
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
    fun `loadModel updates uiState with correct size`() = runTest {
        val viewModel = GameViewModel()
        
        // Mock data
        val size = 3
        val zones = arrayOf(Zone(Zone.Type.ADD, 5, 0))
        val grid = Array(size) { Array(size) { GridCell(1, zones[0]) } }
        val model = KenKenModel(size, zones, grid, false)

        viewModel.loadModel(model)

        assertEquals(size, viewModel.uiState.value.size)
    }
}
