/*
 * SaveManager.kt: Game save/load persistence management
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 */

package org.yegie.orthogon.data

import android.content.Context
import android.content.SharedPreferences
import com.google.gson.Gson
import org.yegie.orthogon.KenKenModel

/**
 * Save slot metadata for display in save/load UI
 */
data class SaveSlotInfo(
    val slotIndex: Int,
    val isEmpty: Boolean,
    val gridSize: Int = 0,
    val difficulty: String = "",
    val timestamp: Long = 0,
    val elapsedSeconds: Long = 0,
    val isSolved: Boolean = false
) {
    val displayName: String
        get() = if (isEmpty) "Empty Slot ${slotIndex + 1}" else "Slot ${slotIndex + 1}: ${gridSize}x${gridSize} $difficulty"

    val formattedTime: String
        get() {
            if (isEmpty) return ""
            val mins = elapsedSeconds / 60
            val secs = elapsedSeconds % 60
            return "%02d:%02d".format(mins, secs)
        }

    val formattedDate: String
        get() {
            if (isEmpty || timestamp == 0L) return ""
            val sdf = java.text.SimpleDateFormat("MM/dd HH:mm", java.util.Locale.US)
            return sdf.format(java.util.Date(timestamp))
        }
}

/**
 * Manages 12 save slots for games in progress.
 * Each slot stores the serialized KenKenModel plus metadata.
 */
class SaveManager(context: Context) {

    companion object {
        const val MAX_SLOTS = 12
        private const val PREFS_NAME = "orthogon_save_slots"
        private const val KEY_MODEL_PREFIX = "slot_model_"
        private const val KEY_SIZE_PREFIX = "slot_size_"
        private const val KEY_DIFF_PREFIX = "slot_diff_"
        private const val KEY_TIME_PREFIX = "slot_time_"
        private const val KEY_ELAPSED_PREFIX = "slot_elapsed_"
        private const val KEY_SOLVED_PREFIX = "slot_solved_"
    }

    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    private val gson = Gson()

    /**
     * Get info for all save slots
     */
    fun getAllSlotInfo(): List<SaveSlotInfo> {
        return (0 until MAX_SLOTS).map { getSlotInfo(it) }
    }

    /**
     * Get info for a specific slot
     */
    fun getSlotInfo(slotIndex: Int): SaveSlotInfo {
        if (slotIndex < 0 || slotIndex >= MAX_SLOTS) {
            return SaveSlotInfo(slotIndex, true)
        }

        val modelJson = prefs.getString("$KEY_MODEL_PREFIX$slotIndex", null)
        if (modelJson.isNullOrEmpty()) {
            return SaveSlotInfo(slotIndex, true)
        }

        return SaveSlotInfo(
            slotIndex = slotIndex,
            isEmpty = false,
            gridSize = prefs.getInt("$KEY_SIZE_PREFIX$slotIndex", 0),
            difficulty = prefs.getString("$KEY_DIFF_PREFIX$slotIndex", "") ?: "",
            timestamp = prefs.getLong("$KEY_TIME_PREFIX$slotIndex", 0),
            elapsedSeconds = prefs.getLong("$KEY_ELAPSED_PREFIX$slotIndex", 0),
            isSolved = prefs.getBoolean("$KEY_SOLVED_PREFIX$slotIndex", false)
        )
    }

    /**
     * Save a game to a specific slot
     */
    fun saveToSlot(
        slotIndex: Int,
        model: KenKenModel,
        difficultyName: String,
        elapsedSeconds: Long
    ): Boolean {
        if (slotIndex < 0 || slotIndex >= MAX_SLOTS) return false

        try {
            val modelJson = gson.toJson(model, KenKenModel::class.java)
            prefs.edit()
                .putString("$KEY_MODEL_PREFIX$slotIndex", modelJson)
                .putInt("$KEY_SIZE_PREFIX$slotIndex", model.size)
                .putString("$KEY_DIFF_PREFIX$slotIndex", difficultyName)
                .putLong("$KEY_TIME_PREFIX$slotIndex", System.currentTimeMillis())
                .putLong("$KEY_ELAPSED_PREFIX$slotIndex", elapsedSeconds)
                .putBoolean("$KEY_SOLVED_PREFIX$slotIndex", model.puzzleWon)
                .apply()
            return true
        } catch (e: Exception) {
            e.printStackTrace()
            return false
        }
    }

    /**
     * Load a game from a specific slot
     */
    fun loadFromSlot(slotIndex: Int): Pair<KenKenModel?, Long> {
        if (slotIndex < 0 || slotIndex >= MAX_SLOTS) return null to 0

        val modelJson = prefs.getString("$KEY_MODEL_PREFIX$slotIndex", null)
        if (modelJson.isNullOrEmpty()) return null to 0

        return try {
            val model = gson.fromJson(modelJson, KenKenModel::class.java)
            model.ensureInitialized()
            val elapsed = prefs.getLong("$KEY_ELAPSED_PREFIX$slotIndex", 0)
            model to elapsed
        } catch (e: Exception) {
            e.printStackTrace()
            null to 0
        }
    }

    /**
     * Delete a save slot
     */
    fun deleteSlot(slotIndex: Int): Boolean {
        if (slotIndex < 0 || slotIndex >= MAX_SLOTS) return false

        prefs.edit()
            .remove("$KEY_MODEL_PREFIX$slotIndex")
            .remove("$KEY_SIZE_PREFIX$slotIndex")
            .remove("$KEY_DIFF_PREFIX$slotIndex")
            .remove("$KEY_TIME_PREFIX$slotIndex")
            .remove("$KEY_ELAPSED_PREFIX$slotIndex")
            .remove("$KEY_SOLVED_PREFIX$slotIndex")
            .apply()
        return true
    }

    /**
     * Find first empty slot, or -1 if all full
     */
    fun findEmptySlot(): Int {
        for (i in 0 until MAX_SLOTS) {
            if (getSlotInfo(i).isEmpty) return i
        }
        return -1
    }
}
