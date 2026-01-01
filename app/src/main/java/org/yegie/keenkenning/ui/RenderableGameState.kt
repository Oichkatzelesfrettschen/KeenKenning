/*
 * RenderableGameState.kt: Platform-agnostic game state for rendering
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 *
 * Provides a unified representation of game state for both:
 * - Legacy KeenView.java (Canvas-based rendering)
 * - Modern GameScreen.kt (Compose-based rendering)
 *
 * This abstraction decouples KeenModel from rendering concerns,
 * enabling independent evolution of both the model and UI layers.
 */

package org.yegie.keenkenning.ui

import org.yegie.keenkenning.KeenModel
import org.yegie.keenkenning.data.GameMode
import org.yegie.keenkenning.perf.PerfTrace

/**
 * Platform-agnostic representation of game state for rendering.
 *
 * This is the contract between the game model and any renderer.
 * Both Canvas-based and Compose-based renderers consume this.
 */
data class RenderableGameState(
    /** Grid dimension (e.g., 4 for a 4x4 grid) */
    val size: Int,

    /** All cells in row-major order [x][y] */
    val cells: List<List<RenderableCell>>,

    /** Zone metadata for clue rendering */
    val zones: List<RenderableZone>,

    /** Currently selected cell, if any */
    val selectedCell: CellCoord?,

    /** Whether the puzzle is solved */
    val isSolved: Boolean,

    /** Whether we're in pencil/notes input mode */
    val isNotesMode: Boolean,

    /** Current game mode (affects display, e.g., MYSTERY hides operations) */
    val gameMode: GameMode,

    /** Whether the grid was ML-generated */
    val isMlGenerated: Boolean,

    /** Elapsed time in seconds (for timer display) */
    val elapsedSeconds: Long,

    /** Difficulty level (0-4) */
    val difficulty: Int
)

/**
 * Cell coordinate for selection tracking.
 */
data class CellCoord(val x: Int, val y: Int)

/**
 * Platform-agnostic cell representation.
 *
 * Contains all information needed to render a single cell
 * without depending on KeenModel internals.
 */
data class RenderableCell(
    /** Cell position */
    val x: Int,
    val y: Int,

    /** Committed digit (1-N), or null if empty */
    val committedValue: Int?,

    /** Pencil marks (notes) - list of N booleans indicating which digits are noted */
    val pencilMarks: List<Boolean>,

    /** Zone this cell belongs to */
    val zoneId: Int,

    /** ML probability hints per digit [0..size], index 0 unused, indices 1-size are probabilities */
    val mlProbabilities: FloatArray?,

    /** Cage borders - which edges have thick zone borders */
    val borders: CageBorders,

    /** Whether this is the zone's anchor cell (shows clue label) */
    val isZoneAnchor: Boolean
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is RenderableCell) return false
        return x == other.x && y == other.y &&
            committedValue == other.committedValue &&
            pencilMarks == other.pencilMarks &&
            zoneId == other.zoneId &&
            borders == other.borders &&
            isZoneAnchor == other.isZoneAnchor
    }

    override fun hashCode(): Int = 31 * x + y
}

/**
 * Cage border flags for a cell.
 *
 * A cell has a thick border on an edge if the adjacent cell
 * belongs to a different zone.
 */
data class CageBorders(
    val top: Boolean = false,
    val bottom: Boolean = false,
    val left: Boolean = false,
    val right: Boolean = false
)

/**
 * Platform-agnostic zone representation.
 */
data class RenderableZone(
    /** Zone identifier */
    val id: Int,

    /** Clue label (e.g., "12+" or "?" for mystery mode) */
    val label: String,

    /** Operation type for semantic styling */
    val operation: ZoneOperation,

    /** Target value */
    val targetValue: Int,

    /** Anchor cell coordinates (where to display the clue) */
    val anchorX: Int,
    val anchorY: Int
)

/**
 * Zone operations for semantic styling.
 * Maps to KeenModel.Zone.Type but decoupled from it.
 */
enum class ZoneOperation {
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    EXPONENT,
    MODULO,
    GCD,
    LCM,
    XOR,
    SINGLE  // Single-cell zones (no operation)
}

/**
 * Transforms KeenModel into RenderableGameState.
 *
 * This is the bridge between the legacy model and modern rendering.
 */
object GameStateTransformer {

    /**
     * Transform a KeenModel into a RenderableGameState.
     *
     * @param model The game model
     * @param isNotesMode Whether the user is in pencil mark mode
     * @param gameMode Current game mode
     * @param elapsedSeconds Timer value
     * @param difficulty Difficulty level
     * @return Platform-agnostic renderable state
     */
    fun transform(
        model: KeenModel,
        isNotesMode: Boolean = false,
        gameMode: GameMode = GameMode.STANDARD,
        elapsedSeconds: Long = 0,
        difficulty: Int = 0
    ): RenderableGameState {
        return PerfTrace.section("GameStateTransformer.transform") {
            val size = model.size
            val zones = transformZones(model, gameMode)
            val cells = transformCells(model, size)
            val selectedCell = if (model.hasActiveCords()) {
                CellCoord(model.activeX.toInt(), model.activeY.toInt())
            } else null

            RenderableGameState(
                size = size,
                cells = cells,
                zones = zones,
                selectedCell = selectedCell,
                isSolved = model.puzzleWon,
                isNotesMode = isNotesMode,
                gameMode = gameMode,
                isMlGenerated = model.wasMlGenerated(),
                elapsedSeconds = elapsedSeconds,
                difficulty = difficulty
            )
        }
    }

    private fun transformZones(model: KeenModel, gameMode: GameMode): List<RenderableZone> {
        val keenZones = model.gameZones ?: return emptyList()
        val grid = model.grid ?: return emptyList()
        val size = model.size

        return keenZones.mapIndexed { index, zone ->
            // Find anchor cell (top-left cell in zone)
            var anchorX = size
            var anchorY = size
            for (x in 0 until size) {
                for (y in 0 until size) {
                    val cell = grid[x][y]
                    if (cell?.zone?.code == zone.code) {
                        if (y < anchorY || (y == anchorY && x < anchorX)) {
                            anchorX = x
                            anchorY = y
                        }
                    }
                }
            }

            val operation = mapOperation(zone.zoneType)
            val label = formatClueLabel(zone, gameMode, operation)

            RenderableZone(
                id = index,
                label = label,
                operation = operation,
                targetValue = zone.expectedValue,
                anchorX = if (anchorX < size) anchorX else 0,
                anchorY = if (anchorY < size) anchorY else 0
            )
        }
    }

    private fun mapOperation(type: KeenModel.Zone.Type): ZoneOperation {
        return when (type) {
            KeenModel.Zone.Type.ADD -> ZoneOperation.ADD
            KeenModel.Zone.Type.MINUS -> ZoneOperation.SUBTRACT
            KeenModel.Zone.Type.TIMES -> ZoneOperation.MULTIPLY
            KeenModel.Zone.Type.DIVIDE -> ZoneOperation.DIVIDE
            KeenModel.Zone.Type.EXPONENT -> ZoneOperation.EXPONENT
            KeenModel.Zone.Type.MODULO -> ZoneOperation.MODULO
            KeenModel.Zone.Type.GCD -> ZoneOperation.GCD
            KeenModel.Zone.Type.LCM -> ZoneOperation.LCM
            KeenModel.Zone.Type.XOR -> ZoneOperation.XOR
        }
    }

    private fun formatClueLabel(
        zone: KeenModel.Zone,
        gameMode: GameMode,
        operation: ZoneOperation
    ): String {
        // Mystery mode hides operations
        if (gameMode == GameMode.MYSTERY) {
            return "?"
        }

        val opSymbol = when (operation) {
            ZoneOperation.ADD -> "+"
            ZoneOperation.SUBTRACT -> "-"
            ZoneOperation.MULTIPLY -> "×"
            ZoneOperation.DIVIDE -> "÷"
            ZoneOperation.EXPONENT -> "^"
            ZoneOperation.MODULO -> "%"
            ZoneOperation.GCD -> "gcd"
            ZoneOperation.LCM -> "lcm"
            ZoneOperation.XOR -> "⊕"
            ZoneOperation.SINGLE -> ""
        }

        return "${zone.expectedValue}$opSymbol"
    }

    private fun transformCells(model: KeenModel, size: Int): List<List<RenderableCell>> {
        val grid = model.grid ?: return emptyList()

        return (0 until size).map { x ->
            (0 until size).map { y ->
                val cell = grid[x][y]
                val zone = cell?.zone

                // Calculate cage borders
                val borders = calculateBorders(grid, x, y, size)

                // Check if this is the zone anchor
                val isAnchor = isZoneAnchor(grid, x, y, size)

                RenderableCell(
                    x = x,
                    y = y,
                    committedValue = cell?.finalGuessValue?.takeIf { it >= 0 },
                    pencilMarks = (0 until size).map { digit ->
                        cell?.guesses?.getOrNull(digit) ?: false
                    },
                    zoneId = zone?.code ?: 0,
                    mlProbabilities = cell?.mlProbabilities,
                    borders = borders,
                    isZoneAnchor = isAnchor
                )
            }
        }
    }

    private fun calculateBorders(
        grid: Array<Array<KeenModel.GridCell?>>,
        x: Int,
        y: Int,
        size: Int
    ): CageBorders {
        val thisZone = grid[x][y]?.zone?.code ?: -1

        val topZone = if (y > 0) grid[x][y - 1]?.zone?.code ?: -1 else -1
        val bottomZone = if (y < size - 1) grid[x][y + 1]?.zone?.code ?: -1 else -1
        val leftZone = if (x > 0) grid[x - 1][y]?.zone?.code ?: -1 else -1
        val rightZone = if (x < size - 1) grid[x + 1][y]?.zone?.code ?: -1 else -1

        return CageBorders(
            top = y == 0 || topZone != thisZone,
            bottom = y == size - 1 || bottomZone != thisZone,
            left = x == 0 || leftZone != thisZone,
            right = x == size - 1 || rightZone != thisZone
        )
    }

    @Suppress("UNUSED_PARAMETER")
    private fun isZoneAnchor(
        grid: Array<Array<KeenModel.GridCell?>>,
        x: Int,
        y: Int,
        size: Int  // Kept for API consistency with other methods
    ): Boolean {
        val thisZone = grid[x][y]?.zone?.code ?: return false

        // Check if any cell above or to the left has the same zone
        for (checkY in 0 until y) {
            if (grid[x][checkY]?.zone?.code == thisZone) return false
        }
        for (checkX in 0 until x) {
            if (grid[checkX][y]?.zone?.code == thisZone) return false
        }

        return true
    }
}

/**
 * Extension to convert RenderableGameState back to UiCell format.
 * This allows gradual migration of GameScreen.kt.
 */
fun RenderableCell.toUiCell(zones: List<RenderableZone>): UiCell {
    val zone = zones.find { it.id == zoneId }
    return UiCell(
        x = x,
        y = y,
        value = committedValue,
        notes = pencilMarks,
        smartHintProbs = mlProbabilities?.toList(),
        zoneId = zoneId,
        isSelected = false, // Set by caller based on selectedCell
        borders = CellBorders(
            top = borders.top,
            bottom = borders.bottom,
            left = borders.left,
            right = borders.right
        ),
        clue = if (isZoneAnchor) zone?.label else null
    )
}
