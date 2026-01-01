/*
 * ClassikNeuralKeenGenerator.java: Stub implementation for Keen Classik (no ML)
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 *
 * This is the Classik flavor stub - always uses algorithmic generation.
 * The full ML implementation is in the Kenning flavor.
 */

package org.yegie.keenkenning;

import android.content.Context;
import android.util.Log;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Stub NeuralKeenGenerator implementation for Keen Classik.
 * Always uses algorithmic generation (no ONNX dependency).
 */
public class ClassikNeuralKeenGenerator implements NeuralKeenGenerator {
    private static final String TAG = "NeuralKeenGen";

    /**
     * Generate a Latin square grid.
     * Classik flavor: Always uses algorithmic backtracking (no ML).
     */
    @Override
    public GenResult generate(Context context, int size) {
        Log.i(TAG, "Classik mode: Using algorithmic generation for size " + size);
        int[] grid = generateAlgorithmicGrid(size);
        if (grid != null) {
            return new GenResult(grid, null);
        }
        return null;
    }

    /** Deprecated signature for compatibility */
    @Override
    public int[] generateGrid(Context context, int size) {
        GenResult res = generate(context, size);
        return res != null ? res.grid : null;
    }

    private int[] generateAlgorithmicGrid(int size) {
        int[] grid = new int[size * size];
        if (fillGrid(grid, size, 0)) {
            return grid;
        }
        return null;
    }

    private boolean fillGrid(int[] grid, int size, int index) {
        if (index == size * size) return true;

        int x = index % size;
        int y = index / size;

        List<Integer> nums = new ArrayList<>();
        for (int i = 1; i <= size; i++) nums.add(i);
        Collections.shuffle(nums);

        for (int num : nums) {
            if (isValidPlacement(grid, size, x, y, num)) {
                grid[index] = num;
                if (fillGrid(grid, size, index + 1)) return true;
                grid[index] = 0;
            }
        }
        return false;
    }

    private boolean isValidPlacement(int[] grid, int size, int x, int y, int num) {
        for (int i = 0; i < x; i++) {
            if (grid[y * size + i] == num) return false;
        }
        for (int i = 0; i < y; i++) {
            if (grid[i * size + x] == num) return false;
        }
        return true;
    }
}
