# Module and Flavor Topology

## Overview
KeenKenning is a single Android app module with two product flavors and two
supporting library modules. Classik is the shared baseline. Kenning adds ML and
narrative features via flavor overrides and the :kenning module.

## Modules

### :app
- Android application module (Compose UI, activities, JNI bridge).
- Hosts product flavors: classik and kenning.
- Entry point for build types, test runners, and native build integration.

### :core
- Shared logic and interfaces for both flavors.
- Should contain pure Kotlin/Java models and non-Android dependencies.

### :kenning
- ML and narrative dependencies (ONNX Runtime, story assets).
- Only wired into the kenning flavor via kenningImplementation.

## Dependencies

- :app -> :core (always)
- :app -> :kenning (kenning flavor only)
- :kenning -> :core (shared contracts)

## Flavors

### classik (baseline)
- Code lives in app/src/main.
- Flavor overrides in app/src/classik are minimal; use only when needed.

### kenning (ML/narrative)
- Overrides in app/src/kenning for flavor-specific behavior.
- Kenning assets live in kenning/src/main/assets.
- Flavor-specific manifest lives in app/src/kenning/AndroidManifest.xml.

## Flavor Services

Flavor-specific wiring should be done via a small factory that exposes
interfaces defined in app/src/main (or :core) and implemented per flavor.
This keeps shared UI code stable and avoids hard dependencies on Kenning-only
classes in the Classik build.

## Decisions (current)
- Keep Classik as the baseline in app/src/main.
- Keep Kenning overrides isolated to app/src/kenning.
- :core holds shared logic; :kenning holds ML + narrative only.
- Target ABIs: armeabi-v7a, arm64-v8a, x86_64; warnings are errors.

## Open Items
- Confirm Kenning assets exist in kenning/src/main/assets and match ML generator expectations.
- Verify flavor overlays remain minimal in app/src/kenning after refactor.
- Keep test runner split: AndroidJUnitRunner for UI, AndroidBenchmarkRunner via -PkeenBenchmark.
