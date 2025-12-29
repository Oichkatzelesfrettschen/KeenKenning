/*
 * SoundManager.java: Audio feedback for game interactions
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning;

import android.content.Context;
import android.media.AudioManager;
import android.view.SoundEffectConstants;

/**
 * Modern Sound Manager using system-native audio feedback.
 */
public class SoundManager {
    private final AudioManager audioManager;

    public SoundManager(Context context) {
        this.audioManager = (AudioManager) context.getSystemService(Context.AUDIO_SERVICE);
    }

    public void playTap() {
        if (audioManager != null) {
            audioManager.playSoundEffect(AudioManager.FX_KEY_CLICK);
        }
    }

    public void playWin() {
        // System success sound
        if (audioManager != null) {
            audioManager.playSoundEffect(AudioManager.FX_FOCUS_NAVIGATION_UP);
        }
    }
}
