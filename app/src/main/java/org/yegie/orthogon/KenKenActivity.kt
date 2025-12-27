/*
 * KenKenActivity.kt: Main game activity with Compose UI
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 */

package org.yegie.orthogon

import android.content.Context
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import com.google.gson.Gson
import org.yegie.orthogon.data.GameMode
import org.yegie.orthogon.ui.GameScreen
import org.yegie.orthogon.ui.GameViewModel

class KenKenActivity : AppCompatActivity() {

    companion object {
        @JvmField val CAN_CONT = "can_continue"
    }

    private val viewModel: GameViewModel by viewModels()

    // Default game data
    private var size = 3
    private var diff = 1
    private var multOnly = 0
    private var seed = 10101L
    private var useAI = false
    private var gameMode = GameMode.STANDARD
    private var continuing = false
    private var gameModel: KenKenModel? = null

    // Names by which to read from saved prefs
    private val SAVE_MODEL = "save_model"
    private val IS_CONT = "is_continuing"

    // Shared prefs file
    private val sharedPref by lazy {
        getSharedPreferences(packageName + "_preferences", Context.MODE_PRIVATE)
    }

    fun getGameData(): Bundle {
        val data = Bundle()
        data.putInt("size", size)
        data.putInt("diff", diff)
        data.putInt("mult", multOnly)
        data.putLong("seed", seed)
        data.putBoolean("useAI", useAI)
        return data
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize SaveManager for save slots
        viewModel.initSaveManager(this)

        // Default to Compose Loading
        setContent {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator()
            }
        }

        continuing = savedInstanceState?.getBoolean(IS_CONT, false) ?: false
    }

    fun returnToMainMenu() {
        finish()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putBoolean(IS_CONT, continuing)
    }

    override fun onResume() {
        super.onResume()

        // If game is already loaded and running, don't reinitialize
        if (gameModel != null && viewModel.uiState.value.size > 0) {
            Log.d("KEEN", "Game already loaded, skipping init")
            return
        }

        val extras = intent.extras
        // Only start new game if explicitly requested via GAME_CONT=false from menu
        val explicitNewGame = extras?.containsKey(MenuActivity.GAME_CONT) == true &&
                              extras.getBoolean(MenuActivity.GAME_CONT) == false

        // Check for saved game in SharedPreferences
        val savedModelJson = sharedPref.getString(SAVE_MODEL, "")
        val hasSavedGame = !savedModelJson.isNullOrEmpty()

        Log.d("KEEN", "onResume: explicitNewGame=$explicitNewGame, hasSavedGame=$hasSavedGame")

        // Restore saved game unless explicitly starting new game from menu
        if (hasSavedGame && !explicitNewGame) {
            try {
                val model = Gson().fromJson(savedModelJson, KenKenModel::class.java)
                model.ensureInitialized()  // Reinitialize transient fields after deserialization
                Log.d("KEEN", "Restoring saved game size=${model.size}")
                runGameModel(model)
                return
            } catch (e: Exception) {
                Log.e("KEEN", "Failed to restore saved game: ${e.message}")
                e.printStackTrace()
                // Fall through to start new game
            }
        }

        // Start new game with parameters from intent
        size = extras?.getInt(MenuActivity.GAME_SIZE, 0) ?: 3
        diff = extras?.getInt(MenuActivity.GAME_DIFF, 0) ?: 1
        multOnly = extras?.getInt(MenuActivity.GAME_MULT, 0) ?: 0
        seed = extras?.getLong(MenuActivity.GAME_SEED, 0L) ?: 0L
        useAI = extras?.getBoolean(MenuActivity.GAME_AI, false) ?: false
        // Parse game mode from intent
        val modeName = extras?.getString(MenuActivity.GAME_MODE) ?: GameMode.STANDARD.name
        gameMode = try {
            GameMode.valueOf(modeName)
        } catch (e: IllegalArgumentException) {
            GameMode.STANDARD
        }

        // Support extended grid sizes (3-16), C layer already handles larger grids
        if (size < 3 || size > 16) {
            Log.e("KEEN", "Got invalid game size, quitting...")
            setResult(RESULT_CANCELED)
            finish()
            return
        }

        Log.d("KEEN", "Starting new game: size=$size")
        runGame()
    }

    override fun onPause() {
        val editor = sharedPref.edit()
        val currentModel = gameModel

        if (currentModel != null) {
            val modelAsString = Gson().toJson(currentModel, KenKenModel::class.java)
            editor.putString(SAVE_MODEL, modelAsString)
            (application as ApplicationCore).setCanCont(!currentModel.puzzleWon)
        } else {
            editor.putString(SAVE_MODEL, "")
            (application as ApplicationCore).setCanCont(false)
        }

        editor.apply()
        super.onPause()
    }

    fun runGameModel(gameModel: KenKenModel) {
        this.gameModel = gameModel
        if (gameModel.wasMlGenerated()) {
             Toast.makeText(this, "ML-Gen Logic: Neural Grid Active", Toast.LENGTH_SHORT).show()
        }

        viewModel.loadModel(gameModel, gameMode = gameMode)
        setContent {
            val uiState by viewModel.uiState.collectAsState()
            if (uiState.size > 0) {
                 GameScreen(viewModel, onMenuClick = { returnToMainMenu() })
            } else {
                 Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                     CircularProgressIndicator()
                 }
            }
        }
        continuing = true
    }

    fun runGame() {
        // Set up reactive Compose UI that observes ViewModel state
        setContent {
            val uiState by viewModel.uiState.collectAsState()
            if (uiState.size > 0) {
                // Game loaded - track model for save/restore
                gameModel = viewModel.getModel()
                continuing = true  // Mark as continuing so onResume doesn't regenerate
                GameScreen(viewModel, onMenuClick = { returnToMainMenu() })
            } else {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator()
                }
            }
        }
        viewModel.startNewGame(this, size, diff, multOnly, seed, useAI, gameMode)
    }
}
