# Installation & Build Requirements

## Prerequisites

### System Requirements
*   **Operating System**: Linux (Verified on CachyOS), macOS, or Windows
*   **RAM**: Minimum 4GB allocated to Gradle daemon (`org.gradle.jvmargs=-Xmx4096m`)

### Software Dependencies
*   **Java Development Kit (JDK)**: Version 21 (Required)
    *   Ensure `JAVA_HOME` points to a valid JDK 21 installation.
*   **Android SDK**:
    *   Compile SDK: 35
    *   Min SDK: 22
    *   Target SDK: 35
*   **Android NDK**: Version 27.2.12479018 (or compatible side-by-side version)
*   **CMake**: Version 3.22.1+
    *   **C Standard**: C11 (Enforced in `CMakeLists.txt`)

## Project Configuration

### Gradle
*   **Build Tool**: Gradle 8.6.0 (via Wrapper)
*   **Kotlin Plugin**: 1.9.22
*   **Android Plugin**: 8.6.0
*   **Warnings as Errors**: Enabled (`allWarningsAsErrors = true`) for strict code quality.

### Native Build (JNI)
*   The project uses CMake to build the native C engine.
*   **ABIs Supported**: `armeabi-v7a`, `arm64-v8a`, `x86`, `x86_64`, `riscv64`
*   **Compiler Flags**: `-Werror -Wall -Wextra` (Strict mode)

## Build Instructions

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd KeenKenning
    ```

2.  **Clean Build**:
    ```bash
    ./gradlew clean
    ```

3.  **Build Debug APK (Kenning Flavor)**:
    ```bash
    ./gradlew assembleKenningDebug
    ```

4.  **Run Unit Tests**:
    ```bash
    ./gradlew testKenningDebugUnitTest
    ```

## Troubleshooting

*   **Build Hangs**: If unit tests hang, ensure `GameViewModel.pauseTimer()` is called in `@After` blocks for tests involving `loadModel`.
*   **Memory Errors**: Verify `gradle.properties` has sufficient heap size (`-Xmx4096m`).
*   **Native Build Failures**: Ensure `ninja` and `cmake` are in your path or managed correctly by the Android SDK Command Line Tools.
