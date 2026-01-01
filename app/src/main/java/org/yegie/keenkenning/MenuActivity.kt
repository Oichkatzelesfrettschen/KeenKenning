/*
 * MenuActivity.kt: Main menu activity with Compose UI
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning

import android.content.Intent
import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.runtime.*
import androidx.core.content.edit
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import org.yegie.keenkenning.data.GameMode
import org.yegie.keenkenning.ui.MenuScreen
import org.yegie.keenkenning.ui.MenuState

/**
 * Modern Compose-based Menu Activity
 */
class MenuActivity : AppCompatActivity() {

    companion object {
        // Intent extras for launching KeenActivity
        const val GAME_SIZE = "gameSize"
        const val GAME_DIFF = "gameDiff"
        const val GAME_MULT = "gameMultOnly"
        const val GAME_MODE = "gameMode"
        const val GAME_SEED = "gameSeed"
        const val GAME_CONT = "contPrev"
        const val GAME_AI = "useAI"

        // SharedPreferences keys (used by ApplicationCore)
        @JvmField val MENU_SIZE = "menuSize"
        @JvmField val MENU_DIFF = "menuDiff"
        @JvmField val MENU_MULT = "menuMult"
        @JvmField val MENU_MODE = "menuMode"
        @JvmField val DARK_MODE = "darkMode"
    }

    private lateinit var app: ApplicationCore

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        app = application as ApplicationCore

        setContent {
            val prefs = getSharedPreferences(packageName + "_preferences", MODE_PRIVATE)
            val lifecycleOwner = LocalLifecycleOwner.current

            val availableModes = GameMode.availableModes()
            val savedModeName = prefs.getString(MENU_MODE, GameMode.STANDARD.name)
            val savedMode = try {
                GameMode.valueOf(savedModeName ?: GameMode.STANDARD.name)
            } catch (e: IllegalArgumentException) {
                GameMode.STANDARD
            }
            val initialMode = if (savedMode in availableModes) savedMode else GameMode.DEFAULT

            var menuState by remember {
                mutableStateOf(
                    MenuState(
                        selectedSize = app.gameSize.coerceIn(3, 16),
                        selectedDifficulty = app.gameDiff,
                        selectedMode = initialMode,
                        multiplicationOnly = initialMode == GameMode.MULTIPLICATION_ONLY,
                        canContinue = app.isCanCont
                    )
                )
            }

            // Refresh state when Activity resumes (e.g., returning from game)
            // This ensures difficulty indicator matches saved game state
            DisposableEffect(lifecycleOwner) {
                val observer = androidx.lifecycle.LifecycleEventObserver { _, event ->
                    if (event == Lifecycle.Event.ON_RESUME) {
                        val currentModeName = prefs.getString(MENU_MODE, GameMode.STANDARD.name)
                        val currentMode = try {
                            GameMode.valueOf(currentModeName ?: GameMode.STANDARD.name)
                        } catch (e: IllegalArgumentException) {
                            GameMode.STANDARD
                        }
                        val clampedMode = if (currentMode in availableModes) {
                            currentMode
                        } else {
                            GameMode.DEFAULT
                        }
                        menuState = menuState.copy(
                            selectedSize = app.gameSize.coerceIn(3, 16),
                            selectedDifficulty = app.gameDiff,
                            selectedMode = clampedMode,
                            multiplicationOnly = clampedMode == GameMode.MULTIPLICATION_ONLY,
                            canContinue = app.isCanCont
                        )
                    }
                }
                lifecycleOwner.lifecycle.addObserver(observer)
                onDispose {
                    lifecycleOwner.lifecycle.removeObserver(observer)
                }
            }

            MenuScreen(
                state = menuState,
                onSizeChange = { size ->
                    menuState = menuState.copy(selectedSize = size)
                    app.gameSize = size
                },
                onDifficultyChange = { diff ->
                    menuState = menuState.copy(selectedDifficulty = diff)
                    app.gameDiff = diff
                },
                onModeChange = { mode ->
                    menuState = menuState.copy(
                        selectedMode = mode,
                        multiplicationOnly = mode == GameMode.MULTIPLICATION_ONLY
                    )
                    // Persist mode selection
                    prefs.edit { putString(MENU_MODE, mode.name) }
                    // Update legacy flag for compatibility
                    app.gameMult = if (mode == GameMode.MULTIPLICATION_ONLY) 1 else 0
                },
                onMultiplicationToggle = { checked ->
                    // Legacy: toggle maps to mode change
                    val newMode = if (checked) GameMode.MULTIPLICATION_ONLY else GameMode.STANDARD
                    menuState = menuState.copy(
                        selectedMode = newMode,
                        multiplicationOnly = checked
                    )
                    app.gameMult = if (checked) 1 else 0
                },
                onStartGame = { startGame(menuState) },
                onContinueGame = { continueGame(menuState) }
            )
        }
    }

    // Note: State refresh on resume is handled by DisposableEffect in setContent

    override fun onPause() {
        app.savePrefs()
        super.onPause()
    }

    private fun startGame(state: MenuState) {
        // Story mode launches the book experience instead of a puzzle
        if (state.selectedMode == GameMode.STORY) {
            val intent = Intent(this, StoryActivity::class.java)
            startActivity(intent)
            return
        }

        val intent = Intent(this, KeenActivity::class.java).apply {
            putExtra(GAME_CONT, false)
            putExtra(GAME_SIZE, state.selectedSize)
            putExtra(GAME_DIFF, state.selectedDifficulty)
            putExtra(GAME_MODE, state.selectedMode.name)
            putExtra(GAME_MULT, state.selectedMode.cFlags and 0x01) // Legacy compat
            putExtra(GAME_SEED, System.currentTimeMillis())
        }
        startActivity(intent)
    }

    @Suppress("UNUSED_PARAMETER")  // State reserved for future use
    private fun continueGame(state: MenuState) {
        val intent = Intent(this, KeenActivity::class.java).apply {
            putExtra(GAME_CONT, true)
        }
        startActivity(intent)
    }
}
