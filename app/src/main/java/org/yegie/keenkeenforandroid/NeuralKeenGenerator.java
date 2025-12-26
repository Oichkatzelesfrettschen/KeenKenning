/*
 * NeuralKeenGenerator.java: ONNX-based neural network puzzle generator
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKeen Contributors
 */

package org.yegie.keenkeenforandroid;

import android.content.Context;
import android.util.Log;
import ai.onnxruntime.*;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Collections;

public class NeuralKeenGenerator {
    private static final String TAG = "NeuralKeenGen";

    public static class GenResult {
        public int[] grid;
        public float[][][][] probabilities; // [Batch][Class][Row][Col]
        
        public GenResult(int[] grid, float[][][][] probs) {
            this.grid = grid;
            this.probabilities = probs;
        }
    }

    public GenResult generate(Context context, int size) {
        Log.i(TAG, "Attempting AI generation for size " + size);
        
        // 1. Try ONNX (Supports sizes 3-9 in this app, model supports up to 20)
        GenResult aiResult = null;
        if (size >= 3 && size <= 9) {
            try {
                aiResult = generateFromModel(context, size);
                if (aiResult != null && isValidLatinSquare(aiResult.grid, size)) {
                     Log.i(TAG, "AI Generation Successful!");
                     return aiResult;
                }
            } catch (Exception e) {
                Log.e(TAG, "AI Generation Failed - falling back to algorithmic", e);
                // Continue to fallback below
            }
        }

        // 2. Fallback: Algorithmic Generation
        Log.i(TAG, "Falling back to algorithmic generation.");
        int[] grid = generateAlgorithmicGrid(size);
        
        // Enhance fallback with AI probabilities if available (even if AI grid was invalid)
        if (grid != null) {
            float[][][][] probs = (aiResult != null) ? aiResult.probabilities : null;
            return new GenResult(grid, probs);
        }
        return null;
    }

    // Deprecated signature
    public int[] generateGrid(Context context, int size) {
        GenResult res = generate(context, size);
        return res != null ? res.grid : null;
    }

    private GenResult generateFromModel(Context context, int size) throws Exception {
        // Copy model and data to cache dir so ORT can find the .data file
        java.io.File cacheDir = context.getCacheDir();
        java.io.File modelFile = new java.io.File(cacheDir, "keen_solver_9x9.onnx");
        java.io.File dataFile = new java.io.File(cacheDir, "keen_solver_9x9.onnx.data"); 
        
        copyAsset(context, "keen_solver_9x9.onnx", modelFile);
        copyAsset(context, "keen_solver_9x9.onnx.data", dataFile);

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        
        // Load from file path
        try (OrtSession session = env.createSession(modelFile.getAbsolutePath(), opts)) {
            // Model now expects fixed 9x9 input of type INT64
            int modelSize = 9;
            long[] inputData = new long[modelSize * modelSize]; // Zero initialized
            
            long[] shape = new long[]{1, modelSize, modelSize}; 
            
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(inputData), shape);
            
            OrtSession.Result result = session.run(Collections.singletonMap("input_grid", inputTensor));
            OnnxTensor outputTensor = (OnnxTensor) result.get(0);
            
            // Output shape: [1, 10, 9, 9] (Batch, Classes, H, W)
            float[][][][] output = (float[][][][]) outputTensor.getValue();
            
            int[] grid = new int[size * size];
            float[][][][] resultProbs = new float[1][size + 1][size][size];

            // Argmax to find best class for each cell
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    int bestClass = 0;
                    float maxVal = -Float.MAX_VALUE;
                    for (int c = 1; c <= size; c++) {
                        float val = output[0][c][y][x];
                        resultProbs[0][c][y][x] = val; // Copy to result
                        
                        if (val > maxVal) {
                            maxVal = val;
                            bestClass = c;
                        }
                    }
                    grid[y * size + x] = bestClass;
                }
            }
            return new GenResult(grid, resultProbs);
        }
    }

    private void copyAsset(Context context, String assetName, java.io.File destFile) throws IOException {
        try (InputStream is = context.getAssets().open(assetName);
             java.io.FileOutputStream fos = new java.io.FileOutputStream(destFile)) {
            byte[] buffer = new byte[4096];
            int read;
            while ((read = is.read(buffer)) != -1) {
                fos.write(buffer, 0, read);
            }
        }
    }

    private boolean isValidLatinSquare(int[] grid, int size) {
        // Simple check: no duplicates in rows/cols
        for (int i = 0; i < size; i++) {
            boolean[] rowSeen = new boolean[size + 1];
            boolean[] colSeen = new boolean[size + 1];
            for (int j = 0; j < size; j++) {
                // Check Row i
                int rVal = grid[i * size + j];
                if (rVal == 0 || rowSeen[rVal]) return false;
                rowSeen[rVal] = true;

                // Check Col i
                int cVal = grid[j * size + i];
                if (cVal == 0 || colSeen[cVal]) return false;
                colSeen[cVal] = true;
            }
        }
        return true;
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

        // Shuffle numbers 1..size for randomness
        java.util.List<Integer> nums = new java.util.ArrayList<>();
        for (int i = 1; i <= size; i++) nums.add(i);
        java.util.Collections.shuffle(nums);

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
        // Check Row
        for (int i = 0; i < x; i++) {
            if (grid[y * size + i] == num) return false;
        }
        // Check Col
        for (int i = 0; i < y; i++) {
            if (grid[i * size + x] == num) return false;
        }
        return true;
    }

}
