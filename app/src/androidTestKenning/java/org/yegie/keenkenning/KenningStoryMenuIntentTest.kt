package org.yegie.keenkenning

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollToNode
import androidx.test.espresso.intent.Intents
import androidx.test.espresso.intent.matcher.IntentMatchers.hasComponent
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.hamcrest.CoreMatchers.allOf
import org.junit.After
import org.junit.Assume
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.yegie.keenkenning.data.GameMode

@RunWith(AndroidJUnit4::class)
class KenningStoryMenuIntentTest {
    @get:Rule
    val composeRule = createAndroidComposeRule<MenuActivity>()

    @Before
    fun setUp() {
        Assume.assumeTrue("Story mode only available in Kenning", BuildConfig.ML_ENABLED)
        Intents.init()
    }

    @After
    fun tearDown() {
        Intents.release()
    }

    @Test
    fun selectingStoryShowsStartGame() {
        val storyLabel = GameMode.STORY.displayName
        composeRule.onNodeWithTag("gameModeList")
            .performScrollToNode(hasText(storyLabel))
        composeRule.waitUntil(10_000) {
            runCatching {
                composeRule.onAllNodesWithText(storyLabel).fetchSemanticsNodes().isNotEmpty()
            }.getOrDefault(false)
        }
        composeRule.onNodeWithText(storyLabel)
            .assertIsDisplayed()
            .performClick()

        composeRule.waitUntil(10_000) {
            composeRule.onAllNodesWithText("START GAME").fetchSemanticsNodes().isNotEmpty()
        }
        composeRule.onNodeWithText("START GAME").assertIsDisplayed().performClick()
        waitForStoryIntent()
    }

    private fun waitForStoryIntent(timeoutMillis: Long = 10_000L) {
        composeRule.waitUntil(timeoutMillis) {
            runCatching {
                Intents.intended(allOf(hasComponent(StoryActivity::class.java.name)))
                true
            }.getOrDefault(false)
        }
    }
}
