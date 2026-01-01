package org.yegie.keenkenning

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assume
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class KenningStoryActivityTest {
    @get:Rule
    val composeRule = createAndroidComposeRule<StoryActivity>()

    @Test
    fun storyCoverNavigationAndExit() {
        Assume.assumeTrue("Story mode only available in Kenning", BuildConfig.ML_ENABLED)

        waitForText("Book")
        composeRule.onNodeWithContentDescription(
            label = "Next page",
            substring = true,
            useUnmergedTree = true
        )
            .assertIsDisplayed()
            .performClick()

        waitForText("Chapter 1")
        composeRule.onNodeWithText("Begin Chapter").assertIsDisplayed()

        composeRule.onNodeWithContentDescription(
            label = "Previous page",
            substring = true,
            useUnmergedTree = true
        )
            .performClick()
        waitForText("Book")

        composeRule.onNodeWithContentDescription(
            label = "Previous page",
            substring = true,
            useUnmergedTree = true
        )
            .performClick()
        composeRule.waitUntil(10_000L) {
            composeRule.activity.isFinishing || composeRule.activity.isDestroyed
        }
    }

    private fun waitForText(
        text: String,
        timeoutMillis: Long = 15_000L
    ) {
        composeRule.waitUntil(timeoutMillis) {
            composeRule.onAllNodesWithText(text, substring = true).fetchSemanticsNodes().isNotEmpty()
        }
    }
}
