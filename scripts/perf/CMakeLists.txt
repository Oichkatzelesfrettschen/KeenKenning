cmake_minimum_required(VERSION 3.22)
project(keen-host-tools C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

set(ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(keen_latin_host
  "${ROOT_DIR}/app/src/main/jni/latin.c"
  "${ROOT_DIR}/app/src/main/jni/random.c"
  "${ROOT_DIR}/app/src/main/jni/malloc.c"
  "${ROOT_DIR}/app/src/main/jni/maxflow_optimized.c"
  "${ROOT_DIR}/app/src/main/jni/tree234.c"
  "${ROOT_DIR}/scripts/ai/host_compat.c"
)

target_include_directories(keen_latin_host PRIVATE "${ROOT_DIR}/app/src/main/jni")
target_compile_options(
  keen_latin_host
  PRIVATE
    -O3
    -fvectorize
    -fslp-vectorize
    -fno-math-errno
    -Wall
    -Wextra
    -Werror
)
target_compile_definitions(keen_latin_host PRIVATE STANDALONE_LATIN_TEST)

enable_testing()
add_test(NAME latin_smoke COMMAND keen_latin_host --seed 1 3)
