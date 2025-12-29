# KeenKenning - Project Memory

## Overview

Android Keen puzzle game (KenKen-style) with two product flavors:
- **Keen Classik**: Traditional mode (3×3-9×9 grids, no ML)
- **Keen Kenning**: Advanced mode (3×3-16×16 grids, ML-enabled)

Architecture:
- **Kotlin/Compose UI** (`app/src/main/java/org/yegie/keenkenning/ui/`)
- **Java integration layer** (`KeenModelBuilder.java`)
- **C backend** (`app/src/main/jni/keen.c`) via JNI
- **ONNX neural solver** (3×3-16×16 grids, Kenning flavor only)

Package: `org.yegie.keenkenning` (with `.classik` or `.kenning` suffix)

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
- **Package naming**: JNI functions follow `Java_org_yegie_keenkenning_<Class>_<Method>`
- **Internal naming**: Classes use "Keen" prefix (trademark-compliant, not "KenKen")
- **AI models**: `latin_solver.onnx` (3×3-16×16, Kenning flavor only)
- **Compose**: UI uses Jetpack Compose in `GameScreen.kt`, `GameViewModel.kt`

## Key Files

| File | Purpose |
|------|---------|
| `app/src/main/jni/keen.c` | Core puzzle generation (C) |
| `app/src/main/jni/keen-android-jni.c` | JNI bridge |
| `KeenModelBuilder.java` | Game state management |
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
3. **Native library**: Must be named `keen-android-jni` (matches CMakeLists.txt)
4. **Trademark**: Use "Keen" prefix for class names (not "KenKen" which is trademarked)

## References

- @docs/SYNTHESIS_REPORT.md - Build harmonization details
- @docs/planning/roadmap.md - Development phases
- @scripts/ai/README.md - Neural model training
- @docs/LATIN_SQUARE_RESEARCH_2025.md - Algorithm research
