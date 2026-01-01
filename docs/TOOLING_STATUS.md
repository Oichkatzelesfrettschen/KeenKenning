# Tooling Status

## Installed tools
- infer: /usr/bin/infer
- perf: /usr/bin/perf (validated via make perf-host)
- FlameGraph: /usr/bin/flamegraph.pl, /usr/bin/stackcollapse-perf.pl, /usr/bin/flamegraph, /usr/bin/cargo-flamegraph

## Askpass helpers
- /home/eirikr/Scripts/sudo-askpass.sh
- /home/eirikr/Documents/dotfiles-scripts/apply-strip-fix-with-askpass.sh
- /usr/bin/unified-askpass
- /usr/bin/qt4-ssh-askpass
- /home/eirikr/Github/Blackhole/scripts/askpass-unified.sh
- /usr/share/windsurf/resources/app/extensions/git/dist/askpass.sh (and related variants)

## Notes and gaps
- LD_PRELOAD=/usr/lib/mklfakeintel.so is set and produces loader warnings in logs.
- Android simpleperf is available in /opt/android-sdk/ndk/27.2.12479018/simpleperf.
- infer runs with C_STD=c2x and -fno-lto; current report is clean after random.c/latin.c/tree234.c fixes.
- valgrind memcheck fails without glibc debuginfo/unstripped ld-linux (see build/valgrind output).
- ctest host build passes with STANDALONE_LATIN_TEST enabled in scripts/perf/CMakeLists.txt.
- gcovr prefers llvm-cov gcov when available (clang-built objects).
