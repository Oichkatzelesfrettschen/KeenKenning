/*
 * KeenModelBuilder.java: JNI bridge for puzzle generation
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2016 Sergey
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning;

import android.util.Log;
import java.util.ArrayList;
import java.util.HashSet;

/**
 * Builder class that bridges Java and Native C (JNI) for Keen puzzle generation.
 * It handles fetching the puzzle definition string from the native layer (either via standard C generation
 * or AI-assisted generation) and parsing it into a KeenModel.
 *
 * JNI Response Format (v2):
 *   Success: "OK:payload_data"
 *   Error:   "ERR:code:message"
 *   Legacy:  "payload_data" (no prefix, for backward compatibility)
 */
public class KeenModelBuilder {

    private static final String PREFIX_OK = "OK:";
    private static final String PREFIX_ERR = "ERR:";

    // Last error message from JNI for debugging/logging
    private String lastJniError = null;

    /**
     * Get the last JNI error message, if any.
     * Useful for debugging when build() returns null.
     */
    public String getLastJniError() {
        return lastJniError;
    }

    /**
     * Parse structured JNI response.
     * @return Payload string on success, null on error
     */
    private String parseJniResponse(String response) {
        if (response == null) {
            lastJniError = "JNI returned null";
            return null;
        }

        if (response.startsWith(PREFIX_ERR)) {
            // Parse error format: "ERR:code:message"
            lastJniError = response.substring(PREFIX_ERR.length());
            Log.w("JNI", "Native error: " + lastJniError);
            return null;
        }

        if (response.startsWith(PREFIX_OK)) {
            // Success format: "OK:payload"
            lastJniError = null;
            return response.substring(PREFIX_OK.length());
        }

        // Legacy format (no prefix) - treat as success
        lastJniError = null;
        return response;
    }

    //instead of rewriting the library in java this uses an NDK to access a modified version
    //of the library in C and then creates a KeenModel based off of it.
    public KeenModel build(android.content.Context context, int size, int diff, int multOnlt, long seed, boolean useAI) {
        return build(context, size, diff, multOnlt, seed, useAI, 0); // Default: MODE_STANDARD = 0
    }

    /**
     * Build with explicit mode flags for extended game modes.
     * @param modeFlags Bit flags from GameMode.cFlags (e.g., 0x04 for ZERO_INCLUSIVE)
     */
    public KeenModel build(android.content.Context context, int size, int diff, int multOnlt, long seed, boolean useAI, int modeFlags)
    {
        String levelAsString = null;
        boolean aiActuallyUsed = false;
        NeuralKeenGenerator.GenResult aiResult = null;

        if (useAI) {
            NeuralKeenGenerator aiGen = new NeuralKeenGenerator();
            aiResult = aiGen.generate(context, size);
            if (aiResult != null && aiResult.grid != null) {
                // Pass the AI grid to C to generate clues and validate
                String rawResponse = getLevelFromAI(size, diff, multOnlt, seed, aiResult.grid, modeFlags);
                levelAsString = parseJniResponse(rawResponse);
                if (levelAsString != null) {
                    aiActuallyUsed = true;
                } else {
                     Log.w("GEN", "AI grid rejected by C solver: " + lastJniError);
                }
            }
        }

        // Fallback to C if AI wasn't used or failed
        if (levelAsString == null) {
            String rawResponse = getLevelFromC(size, diff, multOnlt, seed, modeFlags);
            levelAsString = parseJniResponse(rawResponse);

            if (levelAsString == null) {
                Log.e("GEN", "Native generation failed: " + lastJniError);
                return null;
            }
        }

        String ZoneData = "";

        Log.d("GEN",levelAsString);

        KeenModel.GridCell[][] cells = new KeenModel.GridCell[size][size];
        HashSet<Integer> diffZones = new HashSet<>();

        for(int i = 0; i < levelAsString.length(); i+=3)
        {
            int dsfCount = Integer.parseInt(levelAsString.substring(i,i+2));
            diffZones.add(dsfCount);

            if(levelAsString.charAt(i+2)==';')
            {

                ZoneData = levelAsString.substring(0,i+3);
                levelAsString = levelAsString.substring(i+3);
                break;
            }
        }

        int zoneCount = diffZones.size();
        diffZones.clear();

        KeenModel.Zone[] zones = new KeenModel.Zone[zoneCount];

        for(int i = 0; i<zoneCount; i++)
        {
            char sym = levelAsString.charAt(i*7);
            int val = Integer.parseInt(levelAsString.substring(i*7+1,i*7+6));
            switch(sym)
            {
                case 'a':
                    zones[i] = new KeenModel.Zone(KeenModel.Zone.Type.ADD,val,i);
                    break;
                case 'm':
                    zones[i] = new KeenModel.Zone(KeenModel.Zone.Type.TIMES,val,i);
                    break;
                case 's':
                    zones[i] = new KeenModel.Zone(KeenModel.Zone.Type.MINUS,val,i);
                    break;
                case 'e':
                    zones[i] = new KeenModel.Zone(KeenModel.Zone.Type.EXPONENT,val,i);
                    break;
                default://divide
                    zones[i] = new KeenModel.Zone(KeenModel.Zone.Type.DIVIDE,val,i);
                    break;
            }
        }

        levelAsString = levelAsString.substring(zoneCount*7);

        class zonePairing
        {
            private int raw;
            private int real;

            private zonePairing(int a, int b)
            {
                raw = a; real = b;
            }
        }

        ArrayList<zonePairing> zonePairingList = new ArrayList<>();
        zonePairingList.add(new zonePairing(0,0));

        int nextZoneIndex = 1;


        for(int i = 0; i<size*size; ++i)
        {

            int val = Integer.parseInt(levelAsString.substring(i,i+1));
            int zone = Integer.parseInt(ZoneData.substring(0,2));
            ZoneData = ZoneData.substring(3);

            boolean exists = false;
            int zoneIndex = 0;

            for(int j = 0; j<zonePairingList.size(); ++j)
            {

                if(zone == zonePairingList.get(j).raw)
                {

                    exists = true;
                    zoneIndex = zonePairingList.get(j).real;
                    break;
                }

            }

            if(!exists)
            {
                zonePairingList.add(new zonePairing(zone,nextZoneIndex));
                zoneIndex = nextZoneIndex;
                nextZoneIndex++;
            }

            int x = i/size;
            int y = i%size;

            cells[x][y] = new KeenModel.GridCell(val,zones[zoneIndex]);
            
            // Inject AI probabilities if available (Note: output is [batch][class][y][x])
            // We only have 1 batch.
            if (aiActuallyUsed && aiResult.probabilities != null) {
                float[] cellProbs = new float[size + 1];
                for (int c = 1; c <= size; c++) {
                    // NOTE: ONNX output shape [Batch, Class, Height(y), Width(x)]
                    // x and y in loop above are i/size and i%size.
                    // Let's verify indexing: i goes 0..size*size.
                    // Usually i = y * width + x or x * height + y?
                    // "int x = i/size; int y = i%size;" implies x is row index?
                    // Let's assume standard row-major for now, or match existing logic.
                    // If x=row, y=col:
                    cellProbs[c] = aiResult.probabilities[0][c][x][y];
                }
                cells[x][y].mlProbabilities = cellProbs;
            }

        }

        return new KeenModel(size,zones,cells, aiActuallyUsed);

    }

    static {
        System.loadLibrary("keen-android-jni");
    }

    @SuppressWarnings("JniMissingFunction") //it exists, studio just does not recognize it...
    public native String getLevelFromC(int size, int diff, int multOnly, long seed, int modeFlags);

    @SuppressWarnings("JniMissingFunction")
    public native String getLevelFromAI(int size, int diff, int multOnly, long seed, int[] grid, int modeFlags);


}
