# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.
cmake_minimum_required(VERSION 3.22)

project(keen-android-jni)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
add_library(
            keen-android-jni
            SHARED
            src/main/jni/keen-android-jni.c
            src/main/jni/dlx.c
            src/main/jni/dsf.c
            src/main/jni/keen.c
            src/main/jni/keen_hints.c
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

# Specifies libraries CMake should link to your target library.
target_link_libraries(
            keen-android-jni
            android
            log
)
