/*
 * MenuActivity.kt: Main menu activity with Compose UI
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 */

package org.yegie.orthogon

import android.content.Intent
import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.runtime.*
import org.yegie.orthogon.data.GameMode
import org.yegie.orthogon.ui.MenuScreen
import org.yegie.orthogon.ui.MenuState

/**
 * Modern Compose-based Menu Activity
 */
class MenuActivity : AppCompatActivity() {

    companion object {
        // Intent extras for launching KenKenActivity
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

            // Load saved game mode, default to STANDARD
            val savedModeName = prefs.getString(MENU_MODE, GameMode.STANDARD.name)
            val savedMode = try {
                GameMode.valueOf(savedModeName ?: GameMode.STANDARD.name)
            } catch (e: IllegalArgumentException) {
                GameMode.STANDARD
            }

            var menuState by remember {
                mutableStateOf(
                    MenuState(
                        selectedSize = app.gameSize.coerceIn(3, 16),
                        selectedDifficulty = app.gameDiff,
                        selectedMode = savedMode,
                        multiplicationOnly = savedMode == GameMode.MULTIPLICATION_ONLY,
                        canContinue = app.isCanCont
                    )
                )
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
                    prefs.edit().putString(MENU_MODE, mode.name).apply()
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

    override fun onResume() {
        super.onResume()
        // Update canContinue state when returning from game
        // Note: State is managed in Compose, so we'd need to trigger recomposition
        // For simplicity, this is handled in onCreate's initial state
    }

    override fun onPause() {
        app.savePrefs()
        super.onPause()
    }

    private fun startGame(state: MenuState) {
        val intent = Intent(this, KenKenActivity::class.java).apply {
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
        val intent = Intent(this, KenKenActivity::class.java).apply {
            putExtra(GAME_CONT, true)
        }
        startActivity(intent)
    }
}
