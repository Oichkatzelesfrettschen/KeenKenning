package org.yegie.keenkenning

import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.Assert.*

@RunWith(AndroidJUnit4::class)
class AiIntegrationTest {
    @Test
    fun testAiGenerationSize4() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val generator = NeuralKeenGenerator()
        
        // Test Size 4 (Supported by Model)
        val result = generator.generate(appContext, 4)
        
        assertNotNull("AI Result should not be null for size 4", result)
        assertNotNull("Grid should not be null", result.grid)
        assertEquals("Grid size should be 16", 16, result.grid.size)
        assertNotNull("Probabilities should not be null", result.probabilities)
        
        // Verify valid Latin Square
        assertTrue("Generated grid should be valid", isValidLatinSquare(result.grid, 4))
    }

    private fun isValidLatinSquare(grid: IntArray, size: Int): Boolean {
        for (i in 0 until size) {
            val row = mutableSetOf<Int>()
            val col = mutableSetOf<Int>()
            for (j in 0 until size) {
                // Check Row
                val rVal = grid[i * size + j]
                if (rVal < 1 || rVal > size || !row.add(rVal)) return false
                
                // Check Col
                val cVal = grid[j * size + i]
                if (cVal < 1 || cVal > size || !col.add(cVal)) return false
            }
        }
        return true;
    }
}