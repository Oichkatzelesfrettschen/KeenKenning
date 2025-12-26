# KeenKeenForAndroid - Project Memory

## Overview

Android KenKen-style puzzle game combining:
- **Kotlin/Compose UI** (`app/src/main/java/org/yegie/keenkeenforandroid/ui/`)
- **Java integration layer** (`KeenModelBuilder.java`, `NeuralKeenGenerator.java`)
- **C backend** (`app/src/main/jni/keen.c`) via JNI
- **ONNX neural solver** (`keen_solver_9x9.onnx` supporting 4x4-9x9 grids)

Package: `org.yegie.keenkeenforandroid` (distinct from original "Keen" for side-by-side install)

## Build

```bash
# Debug build (uses build.sh to enforce JDK 21)
./scripts/build.sh assembleDebug

# Or with explicit JAVA_HOME
JAVA_HOME=/usr/lib/jvm/java-21-openjdk ./gradlew assembleDebug

# Lint check
./gradlew lintDebug
```

Requirements: JDK 21, Android SDK, NDK 27.x

## Standards

- **Warnings as errors**: Lint `warningsAsErrors = true` in `app/build.gradle`
- **Package naming**: JNI functions follow `Java_org_yegie_keenkeenforandroid_<Class>_<Method>`
- **AI models**: Production model is `keen_solver_9x9.onnx` (not legacy `tiny_keen_solver.onnx`)
- **Compose**: UI uses Jetpack Compose in `GameScreen.kt`, `GameViewModel.kt`

## Key Files

| File | Purpose |
|------|---------|
| `app/src/main/jni/keen.c` | Core puzzle generation (C) |
| `app/src/main/jni/keen-android-jni.c` | JNI bridge |
| `NeuralKeenGenerator.java` | ONNX Runtime inference |
| `KeenModelBuilder.java` | Game state management |
| `GameScreen.kt` | Compose UI with quantum visualization |
| `scripts/ai/train_massive_model.py` | Production model training |

## Common Issues

1. **JDK version mismatch**: Gradle 8.6 requires JDK 21, not 25
2. **Model name confusion**: Code uses `keen_solver_9x9.onnx`, not legacy `tiny_keen_solver.onnx`
3. **JNI naming**: Function names must match exact package path with underscores

## References

- @docs/SYNTHESIS_REPORT.md - Build harmonization details
- @docs/planning/roadmap.md - Development phases
- @scripts/ai/README.md - Neural model training
- @docs/LATIN_SQUARE_RESEARCH_2025.md - Algorithm research
