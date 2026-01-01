# 0001 Flavor Topology and Module Isolation

Status: accepted

## Context
KeenKenning ships two flavors: Classik (baseline) and Kenning (ML + narrative).
We need shared logic without ML dependencies in the Classik build.

## Decision
- Classik is the baseline in app/src/main.
- Kenning overrides live in app/src/kenning.
- Shared logic and interfaces live in :core.
- ML and narrative live in :kenning and are wired via kenningImplementation.
- Kenning assets live in kenning/src/main/assets.
- FlavorServices provides per-flavor wiring.

## Consequences
- Classik stays lean and free of ML dependencies.
- Kenning remains isolated and easy to reason about.
- Asset and manifest overlays must be verified after refactors.
