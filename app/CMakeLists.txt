# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

project(keen-android-jni)

# Strict warnings - treat warnings as errors for quality
# Note: Some warnings from upstream Simon Tatham code are unavoidable:
#   - unused-parameter: callback signatures have unused params
#   - sign-compare: size_t vs int comparisons
#   - missing-field-initializers: struct partial init
#   - unused-function: upstream funcs not used in Android port
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-parameter -Wno-sign-compare")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-missing-field-initializers")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library(
            # Sets the name of the library.
            # This has to match the call like:
            #   System.loadLibrary("keen-android-jni");
            #
            keen-android-jni

            # Sets the library as a shared library.
            #
            SHARED

            # Provides a relative path to your source file(s).
            #
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

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries(
            # Specifies the target library.
            #
            keen-android-jni
)
