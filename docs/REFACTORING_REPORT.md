# Refactoring Report: C23/C++23 Modernization

## 1. Executive Summary
The `KeenKenning` native codebase has been successfully refactored to adhere to **ISO/IEC 9899:2023 (C23)** and **ISO/IEC 14882:2023 (C++23)** standards. The build system now enforces strict compliance with "Warnings as Errors" enabled, ensuring high code quality and type safety.

## 2. Key Changes

### 2.1 Language Standards
*   **Boolean Logic**: All instances of legacy macros `TRUE` (1) and `FALSE` (0) have been replaced with the standard `bool` type and `true`/`false` keywords.
*   **Null Pointers**: All `NULL` macros have been replaced with the `nullptr` keyword.
*   **Attributes**: Unused functions and parameters are now explicitly marked with the standard `[[maybe_unused]]` attribute instead of GNU-specific extensions where possible, or positioned correctly for compatibility.

### 2.2 Build System
*   **CMake**: Upgraded `CMakeLists.txt` to use target-specific properties (`set_target_properties`) for standard enforcement (`C_STANDARD 23`, `C_EXTENSIONS OFF`).
*   **Gradle**: Configured `build.gradle` to pass strict compiler flags (`-Werror`, `-Wall`, `-Wextra`, `-Wconversion`, `-Wshadow`, etc.) to the NDK build.

### 2.3 Code Quality & Safety
*   **Implicit Conversions**: Explicit casts to `size_t`, `int`, and `char` were added to resolve `-Wsign-conversion` and `-Wimplicit-int-conversion` errors, particularly in memory allocation macros (`snewn`, `sresize`) and string manipulation.
*   **Shadowing**: Renamed local variables (e.g., `p` to `p_val`) to avoid shadowing variables in outer scopes.
*   **Unused Code**: Identified and suppressed unused legacy functions (derived from the original desktop implementation) using standard attributes to maintain a clean build without deleting potentially useful reference code.

## 3. Verification
*   **Build Status**: `BUILD SUCCESSFUL` via `./gradlew assembleDebug`.
*   **Compiler**: Clang (Android NDK r27).
*   **Architecture**: Verified on `arm64-v8a`.

## 4. Recommendations
*   **Continuous Integration**: Ensure the CI pipeline uses the same strict flags to prevent regression.
*   **Legacy Code**: Consider a future pass to remove the `[[maybe_unused]]` functions if it is confirmed they will never be reinstated for the Android version.
