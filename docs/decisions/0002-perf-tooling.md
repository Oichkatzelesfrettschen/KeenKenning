# 0002 Performance Tooling Pipeline

Status: accepted

## Context
We need repeatable performance analysis for native hot paths and stable
profiling workflows for both host tooling and Android builds.

## Decision
- Keep host perf workflows in scripts/perf (perf, flamegraph, gcovr, valgrind, infer).
- Provide a host CMake + CTest harness for smoke validation.
- Use Gradle properties to toggle sanitizers, coverage, PGO, and BOLT relocs.
- Use android_pgo.sh to wrap, pull, and merge device profiles.

## Consequences
- Perf workflows are optional and safe for day-to-day development.
- Native instrumentation can be enabled per build without code changes.
- Android PGO requires device runs and manual workload execution.
