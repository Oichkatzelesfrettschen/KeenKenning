# Keen Kenning Execution Plan (Granular)

## Architecture and Module Boundaries
1. [x] Confirm :core and :kenning modules in settings.gradle.
2. [x] Confirm :app depends on :core and kenningImplementation on :kenning.
3. [x] Keep Classik baseline in app/src/main; Kenning overrides only in app/src/kenning.
4. [x] Kenning assets live in kenning/src/main/assets.
5. [x] Validate ONNX/narrative asset names match code usage (latin_solver optional fallback).
6. [x] Confirm Classik story gating shows stub (no crash).

## Emulator and Instrumented Tests
7. [x] Headless emulator boot script.
8. [x] Readiness check with unlock and package service probe.
9. [x] Keepalive + logcat capture tasks with test-window timestamps.
10. [x] Runner split: UI runner default, benchmark runner via -PkeenBenchmark.
11. [x] Re-run connectedKenningDebugAndroidTest after version upgrades (split batches + Kenning-only tests).
12. [x] Re-run connectedClassikDebugAndroidTest after version upgrades.

## Performance and Instrumentation
13. [x] C23/C++23 + Werror + sanitizer/coverage/PGO toggles wired.
14. [x] Host perf scripts (perf/flamegraph/gcovr/valgrind/infer).
15. [x] Host CMake + CTest harness for native smoke tests.
16. [x] Android PGO collection script (wrap/pull/merge).
17. [x] Add structured perf logging hooks for cross-flavor comparison.

## Benchmarks and Telemetry
18. [x] AndroidX Benchmark scaffolding for parse/layout/solver.
19. [x] Memory/GPU metrics hooks (dumpsys gfxinfo + meminfo).
20. [x] Run benchmarks under AndroidBenchmarkRunner and archive results.

## UI Automation and Crash Triage
21. [x] Story entry/exit smoke test (Kenning).
22. [x] Activity recreate crash repro test.
23. [x] Move Kenning-only androidTests to app/src/kenningAndroidTest.
24. [x] Add full puzzle flow smoke test (launch -> input -> solve).

## Build System and Versions
25. [x] Update AGP/Kotlin/Gradle/Compose versions to latest stable.
26. [x] Enable Gradle build cache, configuration cache, and parallel builds.

## Documentation
27. [x] Add module topology doc (docs/ARCHITECTURE_MODULES.md).
28. [x] Add ADRs for flavor split, ML isolation, and perf tooling.
29. [x] Update roadmap and synthesis docs with current decisions.
30. [x] Update perf/tooling docs with runner split and host tool notes.
31. [x] Update TODO master and task tracker alignment.
