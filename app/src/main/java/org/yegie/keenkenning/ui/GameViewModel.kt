/*
 * GameViewModel.kt: ViewModel for game state management
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning.ui

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import org.yegie.keenkenning.KeenModel
import org.yegie.keenkenning.data.GameMode
import org.yegie.keenkenning.data.PuzzleRepository
import org.yegie.keenkenning.data.PuzzleRepositoryImpl
import org.yegie.keenkenning.data.SaveManager
import org.yegie.keenkenning.data.SaveSlotInfo
import org.yegie.keenkenning.data.StoryChapter
import org.yegie.keenkenning.data.StoryManager
import org.yegie.keenkenning.data.UserStatsManager

class GameViewModel : ViewModel() {
    private val _uiState = MutableStateFlow(GameUiState())
    val uiState: StateFlow<GameUiState> = _uiState.asStateFlow()

    private val repository: PuzzleRepository = PuzzleRepositoryImpl()
    private var keenModel: KeenModel? = null
    private var saveManager: SaveManager? = null
    private var statsManager: UserStatsManager? = null
    private var storyManager: StoryManager? = null
    private var currentStoryChapter: StoryChapter? = null

    // Timer management
    private var timerJob: Job? = null
    private var gameStartTime: Long = 0
    private var accumulatedTime: Long = 0  // For pause/resume

    // Game metadata (set from activity)
    private var currentDifficulty: Int = 1
    private var currentIsMlGenerated: Boolean = false
    private var currentGameMode: GameMode = GameMode.STANDARD

    fun initSaveManager(context: Context) {
        if (saveManager == null) {
            saveManager = SaveManager(context.applicationContext)
        }
        if (statsManager == null) {
            statsManager = UserStatsManager(context.applicationContext)
        }
        if (storyManager == null) {
            storyManager = StoryManager(context.applicationContext)
        }
    }

    fun startNewGame(
        context: Context,
        size: Int,
        diff: Int,
        multOnly: Int,
        seed: Long,
        useAI: Boolean,
        gameMode: GameMode = GameMode.STANDARD
    ) {
        // Adaptive mode: use stats-based difficulty recommendation
        val effectiveDiff = if (gameMode == GameMode.ADAPTIVE) {
            statsManager?.getRecommendedDifficulty(size) ?: diff
        } else {
            diff
        }

        // Store metadata before loading
        currentDifficulty = effectiveDiff
        currentIsMlGenerated = useAI
        currentGameMode = gameMode

        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true) }
            val model = repository.generatePuzzle(context, size, effectiveDiff, multOnly, seed, useAI, gameMode)
            loadModel(model, effectiveDiff, isMlGenerated = useAI, gameMode = gameMode)
        }
    }

    /**
     * Load a model with optional elapsed time restoration.
     *
     * @param model The KeenModel to load
     * @param difficulty Difficulty level (0-4)
     * @param isMlGenerated Whether the puzzle was ML-generated
     * @param gameMode The current game mode
     * @param preservedElapsedSeconds If provided, restore this elapsed time instead of resetting.
     *                                 Use this when resuming a saved game.
     */
    fun loadModel(
        model: KeenModel,
        difficulty: Int = currentDifficulty,
        isMlGenerated: Boolean = currentIsMlGenerated,
        gameMode: GameMode = currentGameMode,
        preservedElapsedSeconds: Long? = null
    ) {
        keenModel = model
        currentDifficulty = difficulty
        currentIsMlGenerated = isMlGenerated
        currentGameMode = gameMode

        // Stop any running timer
        stopTimer()

        // Restore or reset timer based on whether we're resuming
        val elapsed = preservedElapsedSeconds ?: 0L
        accumulatedTime = elapsed

        _uiState.update {
            it.copy(
                showVictoryAnimation = false,
                victoryAnimationComplete = false,
                elapsedTimeSeconds = elapsed,
                timerRunning = false,
                difficulty = difficulty,
                difficultyName = getDifficultyName(difficulty),
                isMlGenerated = isMlGenerated,
                gameMode = gameMode
            )
        }
        refreshState()

        // Start timer (for new game it starts at 0, for resumed game it continues from preserved time)
        if (!model.puzzleWon) {
            startTimer()
        }
    }

    private fun getDifficultyName(diff: Int): String = when (diff) {
        0 -> "Easy"
        1 -> "Normal"
        2 -> "Hard"
        3 -> "Insane"    // Maps to C's DIFF_EXTREME - hard but logically solvable
        4 -> "Ludicrous" // Maps to C's DIFF_UNREASONABLE - may require guessing
        else -> "Unknown"
    }

    // Timer functions
    private fun startTimer() {
        if (timerJob?.isActive == true) return
        if (_uiState.value.isSolved) return

        gameStartTime = System.currentTimeMillis()
        _uiState.update { it.copy(timerRunning = true) }

        timerJob = viewModelScope.launch {
            while (true) {
                delay(1000)
                val elapsed = accumulatedTime + (System.currentTimeMillis() - gameStartTime) / 1000
                _uiState.update { it.copy(elapsedTimeSeconds = elapsed) }
            }
        }
    }

    private fun stopTimer() {
        if (timerJob?.isActive == true) {
            accumulatedTime += (System.currentTimeMillis() - gameStartTime) / 1000
        }
        timerJob?.cancel()
        timerJob = null
        _uiState.update { it.copy(timerRunning = false) }
    }

    fun pauseTimer() {
        stopTimer()
    }

    fun resumeTimer() {
        if (!_uiState.value.isSolved) {
            startTimer()
        }
    }

    fun getModel(): KeenModel? = keenModel

    /**
     * Get the current elapsed time for saving.
     * Includes accumulated time plus current session time.
     */
    fun getElapsedTimeForSave(): Long {
        return if (timerJob?.isActive == true) {
            accumulatedTime + (System.currentTimeMillis() - gameStartTime) / 1000
        } else {
            accumulatedTime
        }
    }

    /**
     * Get the current game mode for save/restore purposes.
     */
    fun getCurrentGameMode(): GameMode = currentGameMode

    private fun refreshState() {
        val model = keenModel ?: return
        val size = model.size

        // Helper to safely get zone ID
        fun getZoneId(x: Int, y: Int): Int {
            if (x < 0 || x >= size || y < 0 || y >= size) return -1
            return model.getCell(x.toShort(), y.toShort()).zone.code
        }

        // Determine anchors for clues (Top-Left-most cell of each zone)
        val zoneAnchors = mutableMapOf<Int, Pair<Int, Int>>()
        for (y in 0 until size) {
            for (x in 0 until size) {
                 val zoneId = getZoneId(x, y)
                 if (!zoneAnchors.containsKey(zoneId)) {
                     zoneAnchors[zoneId] = x to y
                 }
            }
        }

        // Create 2D list of UiCells
        val uiCells = List(size) { x ->
            List(size) { y ->
                val cell = model.getCell(x.toShort(), y.toShort())
                val currentZoneId = cell.zone.code
                
                val borders = CellBorders(
                    top = getZoneId(x, y - 1) != currentZoneId,
                    bottom = getZoneId(x, y + 1) != currentZoneId,
                    left = getZoneId(x - 1, y) != currentZoneId,
                    right = getZoneId(x + 1, y) != currentZoneId
                )

                val notesList = mutableListOf<Boolean>()
                for (g in cell.guesses) {
                    notesList.add(g)
                }
                
                // Convert float[] to List<Float> for UI if present
                val probs = if (cell.mlProbabilities != null) {
                    val list = mutableListOf<Float>()
                    // mlProbabilities is [0..N], 0 unused.
                    // We want list indices 0..N-1 matching numbers 1..N
                    for (i in 1 until cell.mlProbabilities.size) {
                        list.add(cell.mlProbabilities[i])
                    }
                    list
                } else null

                val isAnchor = zoneAnchors[currentZoneId] == (x to y)
                // Mystery mode: show only the target value, hide operation
                val clue = if (isAnchor) {
                    val fullClue = cell.zone.toString()
                    if (currentGameMode == GameMode.MYSTERY) {
                        // Strip operation suffix (e.g., "20 +" → "20?")
                        fullClue.replace(Regex(" [+\\-x/]$"), "?")
                    } else {
                        fullClue
                    }
                } else null

                UiCell(
                    x = x,
                    y = y,
                    value = if (cell.finalGuessValue == -1) null else cell.finalGuessValue,
                    notes = notesList,
                    smartHintProbs = probs,
                    zoneId = currentZoneId,
                    isSelected = (model.activeX.toInt() == x && model.activeY.toInt() == y),
                    borders = borders,
                    clue = clue
                )
            }
        }

        val uiZones = model.gameZones.map { zone ->
             UiZone(
                 id = zone.code,
                 label = zone.toString(),
                 color = 0xFFFFFFFF 
             )
        }

        _uiState.update { 
            it.copy(
                size = size,
                cells = uiCells,
                zones = uiZones,
                activeCell = if (model.activeX >= 0) (model.activeX.toInt() to model.activeY.toInt()) else null,
                isSolved = model.puzzleWon,
                isInputtingNotes = !model.finalGuess,
                isLoading = false
            )
        }
    }
    
    fun onCellClicked(x: Int, y: Int) {
        val model = keenModel ?: return
        
        // Double tap logic: if already selected, toggle note mode
        if (model.activeX.toInt() == x && model.activeY.toInt() == y) {
            model.toggleFinalGuess()
        } else {
            model.setActiveX(x.toShort())
            model.setActiveY(y.toShort())
        }
        refreshState()
    }
    
    fun onInput(number: Int) {
        val model = keenModel ?: return
        val x = model.activeX
        val y = model.activeY
        
        if (x < 0 || y < 0) return 
        
        // Add current state to undo stack before modifying
        model.addCurToUndo(x, y)
        
        if (model.finalGuess) {
             model.clearGuesses(x, y) // Clear notes when entering value
             model.setCellFinalGuess(x, y, number)
        } else {
             model.clearFinal(x, y) // Clear value when entering notes
             model.addToCellGuesses(x, y, number)
        }
        
        model.puzzleWon()
        refreshState()

        // Trigger victory animation if solved
        if (model.puzzleWon && !_uiState.value.showVictoryAnimation) {
            stopTimer()  // Stop timer on victory
            // Record stats for adaptive difficulty (Phase 4b)
            statsManager?.recordPuzzleSolved(
                gridSize = model.size,
                solveTimeSeconds = _uiState.value.elapsedTimeSeconds,
                hintsUsed = _uiState.value.hintsUsed,
                difficulty = currentDifficulty
            )
            // Record story mode progress (Phase 4c)
            if (currentGameMode == GameMode.STORY) {
                recordStoryPuzzleComplete()
            }
            _uiState.update { it.copy(showVictoryAnimation = true) }
        }
    }

    fun onUndo() {
        val model = keenModel ?: return
        model.undoOneStep()
        refreshState()
    }
    
    fun toggleNoteMode() {
        val model = keenModel ?: return
        model.toggleFinalGuess()
        refreshState()
    }
    
    fun toggleSmartHints() {
        _uiState.update { it.copy(showSmartHints = !it.showSmartHints) }
    }
    
    fun toggleInfoDialog() {
        _uiState.update { it.copy(showInfoDialog = !it.showInfoDialog) }
    }

    fun toggleSettingsDialog() {
        _uiState.update { it.copy(showSettingsDialog = !it.showSettingsDialog) }
    }

    fun toggleDarkTheme() {
        _uiState.update { it.copy(darkTheme = !it.darkTheme) }
    }

    fun setFontScale(scale: Float) {
        _uiState.update { it.copy(fontScale = scale.coerceIn(0.8f, 1.5f)) }
    }

    fun toggleShowTimer() {
        _uiState.update { it.copy(showTimer = !it.showTimer) }
    }

    // Immersive mode management
    private var uiAutoHideJob: Job? = null
    private val uiAutoHideDelay = 3000L  // 3 seconds

    fun toggleImmersiveMode() {
        val newImmersive = !_uiState.value.immersiveMode
        _uiState.update { it.copy(immersiveMode = newImmersive, uiVisible = true) }
        if (newImmersive) {
            startUiAutoHideTimer()
        } else {
            cancelUiAutoHideTimer()
        }
    }

    fun showUi() {
        _uiState.update { it.copy(uiVisible = true) }
        if (_uiState.value.immersiveMode) {
            startUiAutoHideTimer()
        }
    }

    fun hideUi() {
        if (_uiState.value.immersiveMode) {
            _uiState.update { it.copy(uiVisible = false) }
        }
    }

    fun toggleUiVisibility() {
        if (_uiState.value.immersiveMode) {
            val newVisible = !_uiState.value.uiVisible
            _uiState.update { it.copy(uiVisible = newVisible) }
            if (newVisible) {
                startUiAutoHideTimer()
            }
        }
    }

    private fun startUiAutoHideTimer() {
        cancelUiAutoHideTimer()
        uiAutoHideJob = viewModelScope.launch {
            delay(uiAutoHideDelay)
            hideUi()
        }
    }

    private fun cancelUiAutoHideTimer() {
        uiAutoHideJob?.cancel()
        uiAutoHideJob = null
    }

    // Call this on any user interaction to reset auto-hide timer
    fun onUserInteraction() {
        if (_uiState.value.immersiveMode && _uiState.value.uiVisible) {
            startUiAutoHideTimer()
        }
    }

    fun setColorblindMode(mode: ColorblindMode) {
        _uiState.update { it.copy(colorblindMode = mode) }
    }

    fun cycleColorblindMode() {
        val modes = ColorblindMode.entries
        val currentIndex = modes.indexOf(_uiState.value.colorblindMode)
        val nextIndex = (currentIndex + 1) % modes.size
        _uiState.update { it.copy(colorblindMode = modes[nextIndex]) }
    }

    fun onVictoryAnimationComplete() {
        _uiState.update { it.copy(victoryAnimationComplete = true) }
    }

    // Keyboard navigation support
    fun moveSelection(dx: Int, dy: Int) {
        val model = keenModel ?: return
        val size = model.size
        val currentX = if (model.activeX >= 0) model.activeX.toInt() else 0
        val currentY = if (model.activeY >= 0) model.activeY.toInt() else 0

        val newX = (currentX + dx).coerceIn(0, size - 1)
        val newY = (currentY + dy).coerceIn(0, size - 1)

        model.setActiveX(newX.toShort())
        model.setActiveY(newY.toShort())
        refreshState()
    }

    fun clearCell() {
        val model = keenModel ?: return
        val x = model.activeX
        val y = model.activeY

        if (x < 0 || y < 0) return

        model.addCurToUndo(x, y)
        model.clearFinal(x, y)
        model.clearGuesses(x, y)
        refreshState()
    }

    // Save/Load dialog toggles
    fun toggleSaveDialog() {
        _uiState.update { it.copy(showSaveDialog = !it.showSaveDialog) }
    }

    fun toggleLoadDialog() {
        _uiState.update { it.copy(showLoadDialog = !it.showLoadDialog) }
    }

    // Get all save slots info
    fun getSaveSlots(): List<SaveSlotInfo> {
        return saveManager?.getAllSlotInfo() ?: emptyList()
    }

    // Save current game to slot
    fun saveToSlot(slotIndex: Int): Boolean {
        val model = keenModel ?: return false
        val manager = saveManager ?: return false

        val result = manager.saveToSlot(
            slotIndex = slotIndex,
            model = model,
            difficultyName = _uiState.value.difficultyName,
            elapsedSeconds = _uiState.value.elapsedTimeSeconds
        )

        if (result) {
            _uiState.update { it.copy(showSaveDialog = false) }
        }
        return result
    }

    // Load game from slot
    fun loadFromSlot(slotIndex: Int): Boolean {
        val manager = saveManager ?: return false
        val (model, elapsed) = manager.loadFromSlot(slotIndex)

        if (model != null) {
            accumulatedTime = elapsed
            loadModel(model)
            _uiState.update { it.copy(showLoadDialog = false) }
            return true
        }
        return false
    }

    // Delete a save slot
    fun deleteSlot(slotIndex: Int): Boolean {
        return saveManager?.deleteSlot(slotIndex) ?: false
    }

    // Phase 4a: Hint/Tutorial Mode
    fun requestHint() {
        val model = keenModel ?: return
        val x = model.activeX.toInt()
        val y = model.activeY.toInt()

        if (x < 0 || y < 0) {
            // No cell selected
            _uiState.update {
                it.copy(
                    showHintDialog = true,
                    currentHint = HintInfo(
                        suggestedDigit = 0,
                        confidence = 0f,
                        reasoning = "Select a cell first to get a hint.",
                        cellX = -1, cellY = -1
                    )
                )
            }
            return
        }

        val cell = model.getCell(x.toShort(), y.toShort())

        // If cell already has a value, no hint needed
        if (cell.finalGuessValue != -1) {
            _uiState.update {
                it.copy(
                    showHintDialog = true,
                    currentHint = HintInfo(
                        suggestedDigit = cell.finalGuessValue,
                        confidence = 1.0f,
                        reasoning = "You already have ${cell.finalGuessValue} here.",
                        cellX = x, cellY = y
                    )
                )
            }
            return
        }

        // Use AI probabilities if available
        val probs = cell.mlProbabilities
        if (probs != null && probs.size > 1) {
            var bestDigit = 1
            var bestProb = 0f
            for (i in 1 until probs.size) {
                if (probs[i] > bestProb) {
                    bestProb = probs[i]
                    bestDigit = i
                }
            }

            val zone = cell.zone
            val reasoning = buildString {
                append("ML suggests $bestDigit")
                if (bestProb >= 0.8f) append(" (high confidence)")
                else if (bestProb >= 0.5f) append(" (moderate confidence)")
                else append(" (uncertain)")
                append(".\nCage target: ${zone.expectedValue} ${zoneOpSymbol(zone.zoneType)}")
            }

            _uiState.update {
                it.copy(
                    showHintDialog = true,
                    currentHint = HintInfo(
                        suggestedDigit = bestDigit,
                        confidence = bestProb,
                        reasoning = reasoning,
                        cellX = x, cellY = y
                    ),
                    hintsUsed = it.hintsUsed + 1
                )
            }
        } else {
            // Fallback: constraint-based hint (cage analysis)
            val zone = cell.zone
            val reasoning = "No ML data available.\nCage target: ${zone.expectedValue} ${zoneOpSymbol(zone.zoneType)}\nTry eliminating impossible values."

            _uiState.update {
                it.copy(
                    showHintDialog = true,
                    currentHint = HintInfo(
                        suggestedDigit = 0,
                        confidence = 0f,
                        reasoning = reasoning,
                        cellX = x, cellY = y
                    ),
                    hintsUsed = it.hintsUsed + 1
                )
            }
        }
    }

    private fun zoneOpSymbol(type: KeenModel.Zone.Type): String = when (type) {
        KeenModel.Zone.Type.ADD -> "+"
        KeenModel.Zone.Type.MINUS -> "-"
        KeenModel.Zone.Type.TIMES -> "×"
        KeenModel.Zone.Type.DIVIDE -> "÷"
        KeenModel.Zone.Type.EXPONENT -> "^"
    }

    fun dismissHint() {
        _uiState.update { it.copy(showHintDialog = false, currentHint = null) }
    }

    fun applyHint() {
        val hint = _uiState.value.currentHint ?: return
        if (hint.suggestedDigit > 0 && hint.cellX >= 0) {
            val model = keenModel ?: return
            model.setActiveX(hint.cellX.toShort())
            model.setActiveY(hint.cellY.toShort())
            model.addCurToUndo(hint.cellX.toShort(), hint.cellY.toShort())
            model.setCellFinalGuess(hint.cellX.toShort(), hint.cellY.toShort(), hint.suggestedDigit)
            model.puzzleWon()
            refreshState()

            if (model.puzzleWon && !_uiState.value.showVictoryAnimation) {
                stopTimer()
                statsManager?.recordPuzzleSolved(
                    gridSize = model.size,
                    solveTimeSeconds = _uiState.value.elapsedTimeSeconds,
                    hintsUsed = _uiState.value.hintsUsed,
                    difficulty = currentDifficulty
                )
                _uiState.update { it.copy(showVictoryAnimation = true) }
            }
        }
        dismissHint()
    }

    // Expose stats for UI display (Phase 4b)
    fun getPlayerStats() = statsManager?.getStats()

    // Phase 4c: Story Mode functions
    fun getStoryChapters() = storyManager?.getAllChapters() ?: emptyList()
    fun getStoryProgress() = storyManager?.getProgress()
    fun isChapterUnlocked(chapterId: String) = storyManager?.isChapterUnlocked(chapterId) ?: false
    fun getCurrentStoryChapter() = currentStoryChapter

    fun startStoryChapter(context: Context, chapterId: String) {
        val manager = storyManager ?: return
        val chapter = manager.getChapter(chapterId) ?: return

        currentStoryChapter = chapter
        manager.setCurrentChapter(chapterId)

        // Start a puzzle using the chapter's settings
        startNewGame(
            context = context,
            size = chapter.gridSize,
            diff = chapter.difficulty,
            multOnly = 0,
            seed = System.currentTimeMillis(),
            useAI = false,
            gameMode = GameMode.STORY
        )
    }

    fun recordStoryPuzzleComplete() {
        val chapter = currentStoryChapter ?: return
        storyManager?.recordPuzzleCompleted(chapter.id)
    }
}