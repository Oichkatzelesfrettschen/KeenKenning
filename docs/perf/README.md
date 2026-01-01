# Performance and Instrumentation Playbook

## Build performance flags
- `./gradlew --parallel --build-cache --configuration-cache --profile <tasks>`
- Gradle properties: `org.gradle.parallel=true`, `org.gradle.caching=true`,
  `org.gradle.configuration-cache=true`

## Local tooling paths (host)
- `perf`: `/usr/bin/perf`
- `flamegraph.pl`: `/usr/bin/flamegraph.pl`
- `stackcollapse-perf.pl`: `/usr/bin/stackcollapse-perf.pl`
- `infer`: `/usr/bin/infer`
- `ctest`: `/usr/bin/ctest`
- `gcovr`: `/usr/bin/gcovr`
- `llvm-cov`: `/usr/bin/llvm-cov`

## Askpass helpers (host)
- `/usr/bin/unified-askpass`
- `/usr/bin/qt4-ssh-askpass`
- `/usr/lib/ssh/x11-ssh-askpass`
- `/usr/lib64/ssh/x11-ssh-askpass`
- `/usr/lib/git-core/git-gui--askpass`
- `/usr/lib/code/extensions/git/dist/askpass.sh` (VS Code)
- `/home/eirikr/dotfiles/config/askpass/unified-askpass`
- `/home/eirikr/Scripts/sudo-askpass.sh`
- `/home/eirikr/Documents/dotfiles-scripts/apply-strip-fix-with-askpass.sh`

## Native instrumentation flags (CMake)
- Sanitizers: `-PkeenSanitizers=address,undefined` (ASan/UBSan)
- Thread sanitizer: `-PkeenSanitizers=thread` (TSan)
- Coverage: `-PkeenCoverage` (sanitizer coverage instrumentation)
- Function tracing: `-PkeenFuncTrace` (adds `-finstrument-functions`)
- LTO (IPO): `-PkeenLto=false` to disable (enabled by default)
- PGO: `-PkeenPgo=generate|use|off` + `-PkeenPgoProfile=/abs/path/profile.profdata` (defaults to `generate`)
- BOLT relocs: `-PkeenBoltRelocs=false` to disable (enabled by default)
- Supported ABIs for sanitizers: `x86_64`, `arm64-v8a` (armeabi-v7a skips)

## Native language standards + warnings
- Pure C builds target `-std=c23` with warnings-as-errors (`-Werror`, `-Wall`, `-Wextra`, etc).
- C++ builds target `-std=c++23` with the same warning gates.

## Scripts
- `scripts/perf/gradle_native_instrument.sh` wraps Gradle with sanitizer flags
- `scripts/perf/emulator_wait_ready.sh` blocks until package service is ready
- `scripts/run_android_tests.sh` runs UI/benchmark tests with keepalive + logcat capture
- `scripts/perf/host_build.sh` builds the host Latin tool with perf/sanitizer/coverage flags
- `scripts/perf/host_perf_record.sh` records perf data for the host tool
- `scripts/perf/host_flamegraph.sh` converts perf data to flamegraph SVG
- `scripts/perf/host_coverage.sh` runs gcovr and writes HTML/XML reports
- `scripts/perf/host_valgrind.sh` runs valgrind memcheck on the host tool
- `scripts/perf/host_infer.sh` runs Infer against the host build
- `scripts/perf/host_pgo.sh` runs the host PGO pipeline (clang + llvm-profdata)
- `scripts/perf/host_bolt.sh` runs the host BOLT pipeline (llvm-bolt + perf2bolt)
- `scripts/perf/host_ctest.sh` builds the host tool via CMake and runs ctest
- `scripts/perf/android_pgo.sh` automates Android PGO wrap, pull, and merge

## Makefile and Gradle wrappers
- Makefile: `make perf-host-build`, `make perf-host`, `make perf-flamegraph`,
  `make perf-coverage`, `make perf-valgrind`, `make perf-infer`, `make perf-pgo`, `make perf-bolt`, `make perf-ctest`,
  `make android-test`, `make android-bench`
- Gradle: `./gradlew perfHostBuild`, `perfHostRecord`, `perfHostFlamegraph`,
  `perfHostCoverage`, `perfHostValgrind`, `perfHostInfer`, `perfHostPgo`, `perfHostBolt`, `perfHostCtest`
  `./gradlew connectedKenningDebugAndroidTest -PkeenBenchmark` (benchmarks)

## Android test runner split
- Default runner: `AndroidJUnitRunner` (UI tests).
- Benchmark runner: `AndroidBenchmarkRunner` via `-PkeenBenchmark` or
  `-Pandroid.testInstrumentationRunner=androidx.benchmark.junit4.AndroidBenchmarkRunner`.
- Wrapper: `scripts/run_android_tests.sh --ui|--bench --task=connectedKenningDebugAndroidTest`.
## Perf and flamegraph (native)
- Host-only (requires host build): `scripts/perf/host_perf_record.sh`
- Flamegraph: `scripts/perf/host_flamegraph.sh` (set `FLAMEGRAPH_DIR` if needed)
- Android device: prefer `simpleperf` or `perfetto` (see below)

## Perfetto and gfxinfo (device)
- `adb shell perfetto -o /data/misc/perfetto-traces/keen.trace -t 10s` with a config
- `adb shell dumpsys gfxinfo org.yegie.keenkenning.kenning` for frame stats
- Structured trace sections are emitted via `PerfTrace` in `PuzzleGenerator`, `PuzzleParser`, and `GameStateTransformer`.

## Coverage and gcovr
- `-PkeenCoverage` enables sanitizer coverage hooks (trace-pc-guard, trace-cmp, etc)
- Host gcovr report: `scripts/perf/host_coverage.sh` (writes to `build/coverage/host`)
- If `--html-details` fails due to pygments/voltron, the script falls back to summary HTML.
- The script prefers `llvm-cov gcov` when available for clang-built objects.

## Valgrind and ctest
- Valgrind memcheck: `scripts/perf/host_valgrind.sh`
- Valgrind needs glibc debuginfo/unstripped `ld-linux` or memcheck fails on `memcmp` redirection.
- ctest: `scripts/perf/host_ctest.sh` (requires CMake + ctest on PATH)

## Infer
- Infer tooling is documented in `~/Documents/Code-Analysis-Tooling/README.md`
- Scripted workflow: `scripts/perf/host_infer.sh` (writes to `build/infer`)
- Infer bundles clang-11; the script forces `C_STD=c2x` and disables LTO for compatibility.

## PGO + BOLT (riced pipeline)
- Host PGO: `make perf-pgo` (builds with clang, trains, merges profile, rebuilds optimized)
- Host BOLT: `make perf-bolt` (builds with relocs, captures perf data, emits BOLT-optimized binary)
- Android PGO:
  - Build instrumented: `scripts/perf/gradle_native_instrument.sh --pgo=generate --lto -- assembleKenningDebug`
  - Run and collect: `scripts/perf/android_pgo.sh --package=org.yegie.keenkenning.kenning --run="shell am start -n org.yegie.keenkenning.kenning/.KeenActivity"`
  - Rebuild with `--pgo=use --pgo-profile=/abs/path/keen.profdata`
- BOLT is host-only for now; Android `.so` binaries do not support a safe BOLT pipeline yet.
