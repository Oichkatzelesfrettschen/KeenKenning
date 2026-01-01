package org.yegie.keenkenning

import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithContentDescription
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class KeenActivityRecreateTest {
    @get:Rule
    val composeRule = createAndroidComposeRule<KeenActivity>()

    @Test
    fun recreateKeepsGridVisible() {
        waitForGrid()
        composeRule.activityRule.scenario.recreate()
        waitForGrid()
    }

    private fun waitForGrid(
        timeoutMillis: Long = 20_000L
    ) {
        composeRule.waitUntil(timeoutMillis) {
            composeRule.onAllNodesWithContentDescription(
                label = "Keen puzzle grid",
                substring = true,
                useUnmergedTree = true
            ).fetchSemanticsNodes().isNotEmpty()
        }
    }
}
