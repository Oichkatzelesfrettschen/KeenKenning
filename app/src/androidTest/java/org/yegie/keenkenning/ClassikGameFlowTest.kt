package org.yegie.keenkenning

import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.SemanticsMatcher
import androidx.compose.ui.test.assertCountEquals
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assume
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ClassikGameFlowTest {
    @get:Rule
    val composeRule = createAndroidComposeRule<MenuActivity>()

    @Before
    fun setUp() {
        Assume.assumeFalse("Classik-only UI flow", BuildConfig.ML_ENABLED)
    }

    @Test
    fun startGameShowsGrid() {
        startGameFromMenu()
        composeRule.onNode(
            hasContentDescriptionContaining("Keen puzzle grid"),
            useUnmergedTree = true
        ).assertIsDisplayed()
    }

    @Test
    fun openGameInfoDialog() {
        startGameFromMenu()
        composeRule.onNode(
            hasContentDescriptionContaining("Open game menu"),
            useUnmergedTree = true
        ).performClick()
        composeRule.onNodeWithText("Game Info").assertIsDisplayed().performClick()
        composeRule.onNodeWithText("How to Play").assertIsDisplayed()
        composeRule.onNodeWithText("Got it!").performClick()
        composeRule.onAllNodesWithText("How to Play").assertCountEquals(0)
    }

    @Test
    fun canEnterNumberInCell() {
        startGameFromMenu()
        composeRule.onNode(
            hasContentDescriptionContaining("Row 1, column 1"),
            useUnmergedTree = true
        ).performClick()
        composeRule.onNode(
            hasContentDescriptionContaining("Number 1"),
            useUnmergedTree = true
        ).performClick()
        waitForNode(
            hasContentDescriptionContaining("Row 1, column 1")
                .and(hasContentDescriptionContaining("contains 1"))
        )
    }

    private fun startGameFromMenu() {
        composeRule.onNodeWithText("START GAME").assertIsDisplayed().performClick()
        waitForNode(hasContentDescriptionContaining("Keen puzzle grid"))
    }

    private fun waitForNode(
        matcher: SemanticsMatcher,
        useUnmergedTree: Boolean = true,
        timeoutMillis: Long = 15000L
    ) {
        composeRule.waitUntil(timeoutMillis) {
            composeRule.onAllNodes(matcher, useUnmergedTree).fetchSemanticsNodes().isNotEmpty()
        }
    }

    private fun hasContentDescriptionContaining(text: String): SemanticsMatcher {
        return SemanticsMatcher("ContentDescription contains \"$text\"") { node ->
            val descriptions = node.config.getOrElse(SemanticsProperties.ContentDescription) { emptyList() }
            descriptions.any { it.contains(text) }
        }
    }
}
