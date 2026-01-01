package org.yegie.keenkenning

import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.SemanticsMatcher
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.test.ext.junit.runners.AndroidJUnit4
import android.util.Log
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.yegie.keenkenning.data.GameMode

@RunWith(AndroidJUnit4::class)
class PuzzleFlowSmokeTest {
    private companion object {
        const val TAG = "PuzzleFlowSmokeTest"
    }

    @get:Rule
    val composeRule = createAndroidComposeRule<MenuActivity>()

    @Test
    fun fullPuzzleFlowFromMenuToGameAndBack() {
        Log.i(TAG, "Selecting mode: Standard")
        selectMode(GameMode.STANDARD.displayName)
        Log.i(TAG, "Starting game from menu")
        composeRule.onNodeWithText("START GAME").assertIsDisplayed().performClick()
        Log.i(TAG, "Waiting for grid")
        waitForNode(hasContentDescriptionContaining("Keen puzzle grid"))

        Log.i(TAG, "Selecting first cell and entering number")
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

        Log.i(TAG, "Opening game info dialog")
        composeRule.onNode(
            hasContentDescriptionContaining("Open game menu"),
            useUnmergedTree = true
        ).performClick()
        composeRule.onNodeWithText("Game Info").assertIsDisplayed().performClick()
        composeRule.onNodeWithText("How to Play").assertIsDisplayed()
        composeRule.onNodeWithText("Got it!").assertIsDisplayed().performClick()

        Log.i(TAG, "Returning to menu")
        composeRule.onNode(
            hasContentDescriptionContaining("Return to main menu"),
            useUnmergedTree = true
        ).performClick()
        composeRule.onNodeWithText("START GAME").assertIsDisplayed()
    }

    private fun selectMode(label: String) {
        val description = "$label mode"
        composeRule.waitUntil(10_000) {
            composeRule.onAllNodes(hasContentDescriptionContaining(description), useUnmergedTree = true)
                .fetchSemanticsNodes()
                .isNotEmpty()
        }
        composeRule.onNode(hasContentDescriptionContaining(description), useUnmergedTree = true)
            .performClick()
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
