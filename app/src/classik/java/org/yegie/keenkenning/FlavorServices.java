/*
 * FlavorServices.java: Classik flavor service factory
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKenning Contributors
 */

package org.yegie.keenkenning;

import android.content.Context;
import org.yegie.keenkenning.data.ClassikNarrativeGenerator;
import org.yegie.keenkenning.data.ClassikSkillModelInference;
import org.yegie.keenkenning.data.ClassikStoryManager;
import org.yegie.keenkenning.data.NarrativeGenerator;
import org.yegie.keenkenning.data.SkillModelInference;
import org.yegie.keenkenning.data.StoryManager;

public final class FlavorServices {
    private FlavorServices() {
    }

    public static NeuralKeenGenerator neuralGenerator() {
        return new ClassikNeuralKeenGenerator();
    }

    public static StoryManager storyManager(Context context) {
        return new ClassikStoryManager(context);
    }

    public static NarrativeGenerator narrativeGenerator(Context context) {
        return ClassikNarrativeGenerator.getInstance(context);
    }

    public static SkillModelInference skillModelInference(Context context) {
        return ClassikSkillModelInference.getInstance(context);
    }
}
