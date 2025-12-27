# Orthogon - Project Memory

## Overview

Android KenKen puzzle game with mathematical branding. Combines:
- **Kotlin/Compose UI** (`app/src/main/java/org/yegie/orthogon/ui/`)
- **Java integration layer** (`KenKenModelBuilder.java`, `NeuralKenKenGenerator.java`)
- **C backend** (`app/src/main/jni/kenken.c`) via JNI
- **ONNX neural solver** (`keen_solver_9x9.onnx` supporting 4x4-9x9 grids)

Package: `org.yegie.orthogon`

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
- **Package naming**: JNI functions follow `Java_org_yegie_orthogon_<Class>_<Method>`
- **Internal naming**: Classes use "KenKen" (the puzzle name), not "Keen"
- **AI models**: Production model is `keen_solver_9x9.onnx`
- **Compose**: UI uses Jetpack Compose in `GameScreen.kt`, `GameViewModel.kt`

## Key Files

| File | Purpose |
|------|---------|
| `app/src/main/jni/kenken.c` | Core puzzle generation (C) |
| `app/src/main/jni/kenken-android-jni.c` | JNI bridge |
| `NeuralKenKenGenerator.java` | ONNX Runtime inference |
| `KenKenModelBuilder.java` | Game state management |
| `GameScreen.kt` | Compose UI with quantum visualization |
| `ui/theme/ColorSystem.kt` | CVD-accessible colors, OLED mode |
| `ui/theme/DesignTokens.kt` | Typography, spacing, battery saver |

## Features

- **OLED Mode**: Pure black backgrounds (#000000) for OLED power savings
- **Battery Saver**: Three-tier animation scaling (OFF/MODERATE/AGGRESSIVE)
- **CVD Accessibility**: Five color profiles (Default, Protan, Deutan, Tritan, Mono)
- **AI Generation**: Optional neural network puzzle generation

## Common Issues

1. **JDK version mismatch**: Gradle 8.6 requires JDK 21, not 25
2. **JNI naming**: Function names must match exact package path with underscores
3. **Native library**: Must be named `kenken-android-jni` (matches CMakeLists.txt)

## References

- @docs/SYNTHESIS_REPORT.md - Build harmonization details
- @docs/planning/roadmap.md - Development phases
- @scripts/ai/README.md - Neural model training
- @docs/LATIN_SQUARE_RESEARCH_2025.md - Algorithm research
