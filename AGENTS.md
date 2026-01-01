# Repository Guidelines

## Project Structure & Module Organization
- `app/`: Android app module. Shared Kotlin/Compose code and resources live in `app/src/main/`.
- `app/src/kenning/` and `app/src/classik/`: product flavors (ML-enabled vs. classic).
- `app/src/main/jni/`: native C/C++ core and JNI bridge.
- `scripts/ai/`: ML data generation and training; ONNX models are deployed to `app/src/kenning/assets/`.
- `app/src/test/` and `app/src/androidTest/`: unit and instrumented tests.
- `docs/`: architecture notes, reports, and screenshots.
- `external/latin-square-toolbox/`: git submodule dependency.

## Build, Test, and Development Commands
- `./gradlew assembleKenningDebug`: build Kenning debug APK.
- `./gradlew assembleKenningRelease`: build release APK.
- `./gradlew testKenningDebugUnitTest`: run JUnit/Robolectric unit tests.
- `./gradlew connectedKenningDebugAndroidTest`: run instrumented tests on device/emulator.
- `./gradlew lintKenningDebug`: run Android Lint (warnings are errors).
- `make build` / `make test` / `make lint`: Makefile wrappers for common Gradle tasks.
- `make tools` or `make train`: build native tools or run ML training pipeline.
Prereqs: JDK 21, Android SDK (API 35), Android NDK 27.x, CMake 3.22+.

## Coding Style & Naming Conventions
- `.editorconfig`: 4-space indent; line length 120 (Kotlin/Java), 100 (C/C++), 88 (Python).
- Kotlin: ktlint + Kotlin conventions; prefer `val`, use data/sealed classes for state.
- C/C++: clang-format (Google style), C23/C++23 with `-Werror`.
- Python: ruff (format + lint), double quotes.
- JNI naming: `Java_org_yegie_keenkenning_<Class>_<Method>`.
- Use the "Keen" prefix in identifiers and UI strings (avoid "KenKen").

## Testing Guidelines
- Unit tests in `app/src/test` (JUnit4 + Robolectric).
- Instrumented tests in `app/src/androidTest` (AndroidX Test + Espresso).
- Example: `./gradlew testKenningDebugUnitTest --tests "*PuzzleParser*"`.

## Commit & Pull Request Guidelines
- Commit style follows Conventional Commits: `feat(scope): ...`, `fix`, `refactor`, `docs`, `chore`.
- PRs should include a clear summary, linked issue (if any), and screenshots for UI changes.
- Run lint and relevant tests; JNI changes should expect the native sanitizer workflow to run.

## Configuration & Secrets
- Keep machine-specific settings in `local.properties` or `~/.gradle/gradle.properties` (e.g., `github.token`).
- Do not commit secrets or local SDK/NDK paths.
