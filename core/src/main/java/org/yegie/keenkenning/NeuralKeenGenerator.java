/*
 * NeuralKeenGenerator.java: Flavor-agnostic ML generation interface
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning;

import android.content.Context;

public interface NeuralKeenGenerator {
    final class GenResult {
        public final int[] grid;
        public final float[][][][] probabilities; // [Batch][Class][Row][Col]

        public GenResult(int[] grid, float[][][][] probabilities) {
            this.grid = grid;
            this.probabilities = probabilities;
        }
    }

    GenResult generate(Context context, int size);

    int[] generateGrid(Context context, int size);
}
