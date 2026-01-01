/*
 * FlavorServices.java: Kenning flavor service factory
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning;

import android.content.Context;
import org.yegie.keenkenning.data.KenningNarrativeGenerator;
import org.yegie.keenkenning.data.KenningSkillModelInference;
import org.yegie.keenkenning.data.KenningStoryManager;
import org.yegie.keenkenning.data.NarrativeGenerator;
import org.yegie.keenkenning.data.SkillModelInference;
import org.yegie.keenkenning.data.StoryManager;

public final class FlavorServices {
    private FlavorServices() {
    }

    public static NeuralKeenGenerator neuralGenerator() {
        return new KenningNeuralKeenGenerator();
    }

    public static StoryManager storyManager(Context context) {
        return new KenningStoryManager(context);
    }

    public static NarrativeGenerator narrativeGenerator(Context context) {
        return KenningNarrativeGenerator.getInstance(context);
    }

    public static SkillModelInference skillModelInference(Context context) {
        return KenningSkillModelInference.getInstance(context);
    }
}
