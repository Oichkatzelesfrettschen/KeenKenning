<!--
  DETERMINISTIC_SEEDING.md: Documentation for reproducible puzzle generation

  SPDX-License-Identifier: GPL-3.0-or-later
  SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKeen Contributors
-->

# Deterministic Seeding in KeenKeen

This document describes the seeding architecture that enables reproducible puzzle
generation. Understanding seeding is essential for debugging, sharing puzzles,
implementing "daily challenge" features, and ensuring test reproducibility.

## Overview

KeenKeen supports deterministic puzzle generation where the same seed value
produces the identical puzzle across sessions and devices (for native generation).
This is achieved through a SHA-based PRNG in the C layer.

## Architecture

```
User Action (New Game)
        │
        ▼
MenuActivity.kt ──────────────────────────────────────────┐
  seed = System.currentTimeMillis()                       │
        │                                                 │
        ▼                                                 │
KeenActivity.kt                                           │
  Receives seed via Intent extras                         │
        │                                                 │
        ▼                                                 │
GameViewModel.kt                                          │
  startNewGame(ctx, size, diff, multOnly, seed, ...)      │
        │                                                 │
        ▼                                                 │
PuzzleRepository.kt                                       │
  generatePuzzle(..., seed, ...)                          │
        │                                                 │
        ▼                                                 │
KeenModelBuilder.java                                     │
  build(..., seed, ...)                                   │
        │                                                 │
        ├──────────── AI Path ────────────┐               │
        │                                 │               │
        ▼                                 ▼               │
getLevelFromC (JNI)             getLevelFromAI (JNI)      │
        │                                 │               │
        ▼                                 ▼               │
random_new(seed)              random_new(seed)            │
  │                           + grid from ONNX            │
  ▼                                 │                     │
SHA-based PRNG                      ▼                     │
  │                           Clue generation             │
  ▼                           uses seeded PRNG            │
Latin Square + Clues                                      │
                                                          │
```

## Seed Flow by Component

### 1. Seed Generation (MenuActivity.kt)

```kotlin
putExtra(GAME_SEED, System.currentTimeMillis())
```

Seeds are generated from the current timestamp when starting a new game.
This provides a unique seed for each game while remaining reproducible
if the exact timestamp is known.

### 2. Seed Storage (KeenActivity.kt)

```kotlin
private var seed = 10101L  // Default fallback

// Saved to instance state:
data.putLong("seed", seed)

// Restored from Intent:
seed = extras?.getLong(MenuActivity.GAME_SEED, 0L) ?: 0L
```

The seed is persisted across configuration changes (screen rotation, etc.)
to ensure game continuity.

### 3. Native PRNG (random.c)

The C layer uses a SHA-based cryptographic PRNG:

```c
random_state *random_new(const char *seed, int len) {
    // SHA-1 based initialization
    SHA_Simple(seed, len, state->seedbuf);
    SHA_Simple(state->seedbuf, 20, state->seedbuf + 20);
    SHA_Simple(state->seedbuf, 40, state->databuf);
    // ...
}
```

Key properties:
- **Portable**: Same seed produces same sequence on all platforms
- **Cryptographic**: High-quality randomness with no observable patterns
- **Efficient**: Uses incremental SHA hashing for next values

### 4. JNI Bridge (keen-android-jni.c)

```c
long lseed = seed;
struct random_state *rs = random_new((char *)&lseed, sizeof(long));
```

The Java `long` is passed directly as bytes to the PRNG initializer.
The endianness is consistent per platform.

## Provider Determinism

The `LatinSquareProvider` interface declares determinism:

| Provider | isDeterministic | Notes |
|----------|-----------------|-------|
| NativeLatinSquareProvider | `true` | SHA-PRNG is perfectly reproducible |
| OnnxLatinSquareProvider | `false` | Neural network sampling has variance |
| HybridLatinSquareProvider | depends | `true` only if both providers are deterministic |

### Native Provider (Fully Deterministic)

Given identical inputs:
- `size`: Grid dimension
- `diff`: Difficulty level
- `multOnly`: Multiplication-only flag
- `seed`: Random seed
- `modeFlags`: Game mode flags

The native provider will **always** produce the **exact same** puzzle:
- Same Latin square solution
- Same cage boundaries
- Same clue targets and operations

### ONNX Provider (Non-Deterministic)

Neural network inference includes:
- Softmax probability sampling
- Temperature-based randomness
- Floating-point precision variations

Even with the same seed, ONNX generation may produce different puzzles.
However, the clue generation phase (which uses the seeded PRNG) is still
deterministic once the Latin square is fixed.

## Testing Seeding

### Verification Test Pattern

```kotlin
@Test
fun `native generation is deterministic`() {
    val seed = 12345L
    val size = 5
    val diff = 1

    // Generate twice with same seed
    val puzzle1 = generateWithSeed(seed, size, diff)
    val puzzle2 = generateWithSeed(seed, size, diff)

    // Must be identical
    assertEquals(puzzle1.zones, puzzle2.zones)
    assertEquals(puzzle1.solution, puzzle2.solution)
}
```

### Snapshot Testing

For regression testing, puzzle snapshots can be stored:

```kotlin
// Expected output for seed=42, size=4, diff=0
val expected = """
    4x4:b5-a2*a2/c7+d1-b6+d5+a3-b3/
"""

@Test
fun `seed 42 produces expected puzzle`() {
    val actual = generateWithSeed(42L, 4, 0)
    assertEquals(expected.trim(), actual.serialized.trim())
}
```

## Use Cases

### 1. Daily Challenges

Use a date-based seed for "Puzzle of the Day":

```kotlin
val today = LocalDate.now()
val dailySeed = today.toEpochDay() * 1_000_000L + size * 100L + diff
```

All users will get the same puzzle for the same day.

### 2. Sharing Puzzles

Puzzles can be shared as compact URLs:

```
keenkeen://puzzle?seed=1735170000000&size=6&diff=2
```

Recipient generates the identical puzzle locally.

### 3. Debugging

When a user reports a bug with a specific puzzle:

```kotlin
// Reproduce exact puzzle state
val bugSeed = 1735170123456L
viewModel.startNewGame(ctx, 5, 2, false, bugSeed, false, GameMode.STANDARD)
```

### 4. Test Reproducibility

Property-based tests use fixed seeds:

```kotlin
(3..9).forEach { size ->
    val puzzle = generateValidLatinSquare(size, seed = 42L + size)
    // Tests are reproducible
}
```

## Edge Cases

### 1. Seed = 0

A seed of 0 is valid but may produce less random initial state.
The default fallback seed (`10101L`) is used if Intent extras are missing.

### 2. Negative Seeds

The `Long` seed can be negative. The byte representation is passed
directly to SHA, so negative values produce valid (different) PRNG states.

### 3. Very Large Seeds

Seeds approaching `Long.MAX_VALUE` work correctly. The SHA hash
normalizes all seed values into a fixed-size state buffer.

## Security Considerations

- **Not Cryptographically Secure for Secrets**: While SHA-based, this PRNG
  is not designed for cryptographic key generation.
- **Predictable Daily Seeds**: Date-based seeds are guessable. Do not use
  for competitive scenarios requiring unpredictability.
- **Seed Leakage**: Avoid logging seeds in production if puzzles should
  remain private until solved.

## Migration Notes

### From v1.x to v2.x

The seeding algorithm remained unchanged. Puzzles generated with the same
seed in v1.x will match v2.x outputs for the native provider.

### Future: Versioned Seeding

If algorithm changes become necessary:

```kotlin
data class VersionedSeed(
    val seed: Long,
    val algorithmVersion: Int = 2
)
```

This ensures old puzzle shares remain reproducible.
