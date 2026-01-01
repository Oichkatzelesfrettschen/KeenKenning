# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.
cmake_minimum_required(VERSION 3.22)

project(keen-android-jni)

set(KEEN_SANITIZERS "" CACHE STRING "Comma-separated sanitizer list (e.g. address,undefined)")
option(KEEN_COVERAGE "Enable sanitizer coverage instrumentation" OFF)
option(KEEN_FUNC_TRACE "Enable -finstrument-functions tracing" OFF)
option(KEEN_LTO "Enable LTO (IPO)" ON)
set(KEEN_PGO "generate" CACHE STRING "PGO mode: generate|use|off")
set(KEEN_PGO_PROFILE "" CACHE STRING "Path to LLVM .profdata for PGO")
option(KEEN_BOLT_RELOCS "Emit relocations for BOLT" ON)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
add_library(
            keen-android-jni
            SHARED
            src/main/jni/keen-android-jni.c
            src/main/jni/dlx.c
            src/main/jni/dsf.c
            src/main/jni/keen.c
            src/main/jni/keen_generate.c
            src/main/jni/keen_hints.c
            src/main/jni/keen_solver.c
            src/main/jni/keen_validate.c
            src/main/jni/latin.c
            src/main/jni/malloc.c
            src/main/jni/maxflow_optimized.c
            src/main/jni/random.c
            src/main/jni/tree234.c
)

# Enforce C23 and C++23 standards on the target
set_target_properties(keen-android-jni PROPERTIES
    C_STANDARD 23
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Explicitly define include directories
target_include_directories(keen-android-jni PRIVATE src/main/jni)
target_compile_options(keen-android-jni PRIVATE -O3 -fvectorize -fslp-vectorize -fno-math-errno)
target_link_options(keen-android-jni PRIVATE -fuse-ld=lld)

if(KEEN_LTO)
    target_compile_options(keen-android-jni PRIVATE -flto=thin)
    target_link_options(keen-android-jni PRIVATE -flto=thin)
endif()

if(KEEN_FUNC_TRACE)
    target_compile_options(keen-android-jni PRIVATE -finstrument-functions)
endif()

if(KEEN_COVERAGE)
    target_compile_options(keen-android-jni PRIVATE -fsanitize-coverage=trace-pc-guard,trace-cmp,trace-div,trace-gep)
endif()

if(KEEN_SANITIZERS)
    if(ANDROID_ABI MATCHES "arm64-v8a|x86_64")
        target_compile_options(
            keen-android-jni
            PRIVATE
            -fsanitize=${KEEN_SANITIZERS}
            -fno-sanitize-recover=all
        )
        target_link_options(keen-android-jni PRIVATE -fsanitize=${KEEN_SANITIZERS})
    else()
        message(STATUS "Skipping sanitizers for ${ANDROID_ABI} (unsupported)")
    endif()
endif()

if(KEEN_PGO STREQUAL "generate")
    target_compile_options(keen-android-jni PRIVATE -fprofile-instr-generate)
    target_link_options(keen-android-jni PRIVATE -fprofile-instr-generate)
elseif(KEEN_PGO STREQUAL "use")
    if(NOT KEEN_PGO_PROFILE)
        message(FATAL_ERROR "KEEN_PGO_PROFILE is required when KEEN_PGO=use")
    endif()
    target_compile_options(keen-android-jni PRIVATE -fprofile-instr-use=${KEEN_PGO_PROFILE})
    target_link_options(keen-android-jni PRIVATE -fprofile-instr-use=${KEEN_PGO_PROFILE})
endif()

if(KEEN_BOLT_RELOCS)
    target_link_options(keen-android-jni PRIVATE -Wl,--emit-relocs)
endif()

# Specifies libraries CMake should link to your target library.
target_link_libraries(
            keen-android-jni
            android
            log
)
