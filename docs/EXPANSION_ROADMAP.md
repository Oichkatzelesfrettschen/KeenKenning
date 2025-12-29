# Keen Kenning Expansion Roadmap

**Date:** December 26, 2025
**Scope:** Comprehensive TODO analysis with research-backed implementation paths
**Goal:** Transform Keen Kenning into an ultrafast, ultrafun Latin square puzzle platform

---

## Executive Summary

This document scopes six identified TODOs against external research from KenKen solvers,
Latin square generators, and Android TV development patterns. Each section includes:
- **Current State**: What exists now
- **Research Findings**: External implementations to learn from
- **Implementation Path**: Concrete steps
- **Submodule Candidates**: Repos to potentially integrate

---

## 1. TODO(operations): New Arithmetic Operations

### Current State
```c
#define C_ADD 0x00000000L  // Addition
#define C_MUL 0x20000000L  // Multiplication
#define C_SUB 0x40000000L  // Subtraction
#define C_DIV 0x60000000L  // Division
#define C_EXP 0x80000000L  // Exponentiation (requires MODE_EXPONENT)
#define CMASK 0xE0000000L  // 3 bits = 8 slots, 5 used, 3 available
```

### Research Findings

**From [CanCan](https://github.com/wpm/CanCan) (Scala):**
- Implements standard 4 operations with constraint propagation
- No modular arithmetic, but extensible design

**From [Kenny](https://github.com/camsteffen/kenny) (Rust):**
- Uses 8 constraint implementations
- Backtracking fallback when propagation fails

**Mathematical Extensions (from LATIN_SQUARE_RESEARCH_2025.md):**
- Modular arithmetic connects to post-quantum cryptography
- GCD/LCM relate to number theory education

### Proposed New Operations

| Code | Operation | Symbol | Example | Difficulty Modifier |
|------|-----------|--------|---------|---------------------|
| `0xA0000000L` | C_MOD | `%` | `7 % 3 = 1` | +1 level |
| `0xC0000000L` | C_GCD | `gcd` | `gcd(12,8) = 4` | +1 level |
| `0xE0000000L` | C_LCM | `lcm` | `lcm(4,6) = 12` | +2 levels |

### Implementation Path

1. **Phase 1: C Backend**
   - Add operation codes to `kenken.c`
   - Extend `clue_to_string()` for symbols
   - Update `combine_clue()` with mod/gcd/lcm logic
   - Add `MODE_NUMBER_THEORY` flag to `kenken_modes.h`

2. **Phase 2: Solver Extensions**
   - GCD propagation: If cage=4, cells ∈ {4,8,12,...} ∩ {1..N}
   - LCM propagation: If cage=12, cells must divide 12
   - MOD requires paired reasoning (a % b = c means a = kb + c)

3. **Phase 3: UI/UX**
   - Add operation toggle in settings
   - Educational tooltips explaining new operations
   - "Research Mode" difficulty tier

### Submodule Candidates
- **[nathanoschmidt/latin-square-toolbox](https://github.com/nathanoschmidt/latin-square-toolbox)**:
  Contains modular arithmetic for cryptographic Latin squares

---

## 2. TODO(large-grids): 16x16 Grid Support

### Current State
```c
#define MAXBLK 6  // Maximum cage size
// Clues use `long` (32-bit on some platforms)
```

### Research Findings

**Overflow Risk Analysis:**
- Maximum product in 6-cell cage: 16^6 = 16,777,216 (fits in 32-bit)
- Maximum product in 8-cell cage: 16^8 = 4,294,967,296 (OVERFLOWS 32-bit!)
- Maximum product in 7-cell cage: 16^7 = 268,435,456 (fits)

**From [PritK99/Latin-Square-Completion](https://github.com/PritK99/Latin-Square-Completion):**
- Uses parallel Tabu Search for large grids
- BFS/DFS become impractical above 12x12

**From [3N4N/csp-latin-square](https://github.com/3N4N/csp-latin-square):**
- Forward checking essential for N > 12
- Arc consistency critical for N = 15+

### Implementation Path

1. **Phase 1: Dynamic MAXBLK**
   ```c
   #define MAXBLK_FOR_SIZE(w) ((w) <= 9 ? 6 : (w) <= 12 ? 5 : 4)
   ```
   - Smaller cages for larger grids = faster generation

2. **Phase 2: 64-bit Clues**
   ```c
   // Change from:
   long *clues;
   // To:
   int64_t *clues;  // or `long long`
   ```
   - Update all clue arithmetic to use 64-bit

3. **Phase 3: Generation Optimization**
   - Implement constraint propagation from [Kenny](https://github.com/camsteffen/kenny)
   - Add parallel candidate enumeration (AI model already handles this)
   - Timeout escalation: try smaller cages if generation stalls

4. **Phase 4: UI Scaling**
   - Smaller fonts for 16x16 grids
   - Pinch-to-zoom for cell selection
   - Mini-map navigation overlay

### Submodule Candidates
- **[latin-square-toolbox](https://github.com/nathanoschmidt/latin-square-toolbox)**:
  DSP mode for fast large-grid generation

---

## 3. TODO(expansion): AI Generation Bypass

### Current State
```c
// In new_game_desc() at kenken.c:899
if (w == 3 && diff > DIFF_NORMAL)
    diff = DIFF_NORMAL;  // Legacy bypass
```

### Research Findings

**Current AI Flow:**
1. `NeuralKenKenGenerator.java` generates Latin square via ONNX
2. Passes to `new_game_desc_from_grid()` in C
3. C generates cages and validates solvability

**From Analysis:**
- AI model (latin_solver.onnx) handles 3x3 through 16x16
- Trained with curriculum learning and constraint-aware loss
- Eliminates exhaustive Latin square search entirely

### Implementation Path

1. **Phase 1: Remove 3x3 Bypass**
   ```c
   // DELETE these lines:
   // if (w == 3 && diff > DIFF_NORMAL)
   //     diff = DIFF_NORMAL;
   ```

2. **Phase 2: AI-First Generation Path**
   ```kotlin
   suspend fun generatePuzzle(size: Int, diff: Difficulty): Puzzle {
       // Try AI first (fast path)
       val aiGrid = neuralGenerator.generate(size)
       if (aiGrid != null) {
           return nativeBuilder.buildFromGrid(aiGrid, diff)
       }
       // Fallback to pure C (slow path)
       return nativeBuilder.generateRandom(size, diff)
   }
   ```

3. **Phase 3: Adaptive Difficulty**
   - If AI grid doesn't meet difficulty target, regenerate
   - Track generation success rates per size/difficulty
   - Auto-adjust parameters based on success rate

4. **Phase 4: Model Expansion**
   - Train 16x16 model using `scripts/ai/train_massive_model.py`
   - Quantize for mobile (INT8)
   - A/B test against pure C generation

---

## 4. TODO(android-tv): D-pad Navigation

### Current State
- Compose UI with touch-only interaction
- No focus management for non-touch devices
- Legacy desktop code removed (had cursor navigation)

### Research Findings

**From [Android Developers - Focus in Compose](https://developer.android.com/develop/ui/compose/touch-input/focus):**
- Use `Modifier.focusable()` for interactive elements
- `FocusRequester` for programmatic focus control
- `focusProperties` for custom navigation

**From [thesauri/dpad-compose](https://github.com/thesauri/dpad-compose):**
- Custom `.dpadFocusable()` modifier pattern
- Handles focus loss on screen transitions
- Grid-based focus traversal

**From [alexzaitsev - Focus as State](https://alexzaitsev.substack.com/p/focus-as-a-state-new-effective-tv):**
- Treat focus as reactive state
- Encapsulate in `FocusRequesterModifiers` class
- Known bug: nested lazy container focus restoration

### Implementation Path

1. **Phase 1: Focus Infrastructure**
   ```kotlin
   @Composable
   fun GameCell(
       row: Int, col: Int,
       onSelect: () -> Unit,
       modifier: Modifier = Modifier
   ) {
       val focusRequester = remember { FocusRequester() }
       Box(
           modifier = modifier
               .focusRequester(focusRequester)
               .focusable()
               .onKeyEvent { event ->
                   when (event.key) {
                       Key.DirectionCenter, Key.Enter -> {
                           onSelect()
                           true
                       }
                       else -> false
                   }
               }
       ) { /* cell content */ }
   }
   ```

2. **Phase 2: Grid Navigation**
   ```kotlin
   Modifier.focusProperties {
       left = if (col > 0) cells[row][col-1] else FocusRequester.Cancel
       right = if (col < size-1) cells[row][col+1] else FocusRequester.Cancel
       up = if (row > 0) cells[row-1][col] else FocusRequester.Cancel
       down = if (row < size-1) cells[row+1][col] else FocusRequester.Cancel
   }
   ```

3. **Phase 3: Number Pad Overlay**
   - D-pad navigates grid
   - Number keys (1-9) enter values directly
   - Long-press center = pencil mode toggle

4. **Phase 4: Chromebook Keyboard**
   - Full keyboard shortcuts (Ctrl+Z undo, etc.)
   - Tab cycles through cages
   - Space = toggle pencil mode

### Submodule Candidates
- **[thesauri/dpad-compose](https://github.com/thesauri/dpad-compose)**:
  Tutorial and reference implementation

---

## 5. TODO(validation): Error Checking Extraction

### Current State
- Validation logic exists in `GameViewModel.kt`
- Original C `check_errors()` was removed with legacy desktop code
- No JNI-exposed validation

### Research Findings

**From [billabob.github.io/kenkensolver](https://billabob.github.io/kenkensolver/):**
- Hierarchical error detection levels
- Basic: Row/column duplicates
- Intermediate: Cage arithmetic violations
- Advanced: Impossible cell candidates

**Validation Categories:**
1. **Immediate**: Duplicate in row/column (Latin square violation)
2. **Cage-level**: Arithmetic doesn't match clue
3. **Deductive**: No valid completion exists (requires solver)

### Implementation Path

1. **Phase 1: Extract C Validation Module**
   Create `kenken_validate.c`:
   ```c
   typedef struct {
       int error_type;  // 0=none, 1=duplicate, 2=cage, 3=unsolvable
       int row, col;    // Error location
       int related[16]; // Related cells (for highlighting)
   } ValidationError;

   int validate_grid(const digit *grid, int w,
                     const struct clues *clues,
                     ValidationError *errors, int max_errors);
   ```

2. **Phase 2: JNI Bridge**
   ```c
   JNIEXPORT jintArray JNICALL
   Java_org_yegie_keenkenning_KenKenModelBuilder_validateGrid(
       JNIEnv *env, jobject instance,
       jintArray gridFlat, jint size);
   ```

3. **Phase 3: Real-time UI Integration**
   ```kotlin
   // In GameViewModel
   private fun validateAndHighlight() {
       val errors = nativeBuilder.validateGrid(currentGrid, size)
       _uiState.update { it.copy(errorCells = errors.toSet()) }
   }
   ```

4. **Phase 4: Error Explanation**
   - Tap error cell = show explanation
   - "This cell conflicts with row 3"
   - "Cage sum is 15, but cells total 17"

---

## 6. TODO(hints): Step-by-Step Hint System

### Current State
- No hint system
- Solver exists in C (`solve_game()` was in removed code)
- Full solution available via `aux` string

### Research Findings

**From [billabob.github.io/kenkensolver](https://billabob.github.io/kenkensolver/) - Technique Hierarchy:**

| Level | Technique | Description |
|-------|-----------|-------------|
| 0-1 | Naked/Hidden Singles | Only one possibility for cell |
| 1.5 | Pointing/Claiming | Cage-region intersection |
| 2.0 | Hidden Subsets | Groups eliminate candidates |
| 2.2 | X-Wing/Swordfish | Fish patterns |
| 3.0 | Region Parity | Even/odd constraints |
| 4.0+ | Cage Combinations | Multi-cage reasoning |

**From [CanCan](https://github.com/wpm/CanCan):**
- `analyze` command provides detailed step-by-step output
- Each elimination logged with reasoning

### Implementation Path

1. **Phase 1: Solver State Exposure**
   Create `kenken_hints.c`:
   ```c
   typedef struct {
       int technique_level;  // 0-5 difficulty
       const char *technique_name;
       int target_cell;      // Cell affected
       int eliminated[16];   // Candidates removed
       char explanation[256];
   } HintStep;

   int get_next_hint(const digit *grid, int w,
                     const struct clues *clues,
                     const unsigned char *pencil_marks,
                     HintStep *hint);
   ```

2. **Phase 2: Technique Implementations**
   - Level 1: Naked singles ("Only 4 fits in R2C3")
   - Level 2: Hidden singles ("4 must go in R2C3 in row 2")
   - Level 3: Cage math ("Sum cage needs 7, only 3+4 works")
   - Level 4: Cross-cage ("If R1C1=5, cage sum impossible")

3. **Phase 3: JNI Bridge**
   ```c
   JNIEXPORT jstring JNICALL
   Java_org_yegie_keenkenning_KenKenModelBuilder_getHint(
       JNIEnv *env, jobject instance,
       jintArray gridFlat, jintArray pencilMarks, jint size);
   ```

4. **Phase 4: UI Integration**
   - "Hint" button in toolbar
   - Animated highlight of affected cells
   - Progressive hints (vague → specific)
   - "Show me" option applies the step

5. **Phase 5: Learning Mode**
   - Explain each technique as it's used
   - Track which techniques user masters
   - Suggest puzzles targeting weak areas

---

## External Repositories Analysis

### Potential Submodules

| Repository | Language | Use Case | Integration Effort |
|------------|----------|----------|-------------------|
| [camsteffen/kenny](https://github.com/camsteffen/kenny) | Rust | Constraint propagation reference | Medium (FFI) |
| [wpm/CanCan](https://github.com/wpm/CanCan) | Scala | Difficulty grading algorithm | Low (port logic) |
| [nathanoschmidt/latin-square-toolbox](https://github.com/nathanoschmidt/latin-square-toolbox) | C | Fast generation (DS/DSP) | Low (C integration) |
| [chrisboyle/sgtpuzzles](https://github.com/chrisboyle/sgtpuzzles) | C/Kotlin | Android reference | Reference only |
| [thesauri/dpad-compose](https://github.com/thesauri/dpad-compose) | Kotlin | TV navigation | Low (copy pattern) |
| [gkaranikas/dancing-links](https://github.com/gkaranikas/dancing-links) | C++ | Algorithm X solver | Medium |
| [PritK99/Latin-Square-Completion](https://github.com/PritK99/Latin-Square-Completion) | Python | Parallel Tabu Search | Reference only |

### Recommended Submodule Additions

```bash
# Latin square generation algorithms
git submodule add https://github.com/nathanoschmidt/latin-square-toolbox.git external/latin-square-toolbox

# Dancing Links for exact cover solving
git submodule add https://github.com/gkaranikas/dancing-links.git external/dancing-links
```

---

## Implementation Priority Matrix

| TODO | Impact | Effort | Priority | Dependencies |
|------|--------|--------|----------|--------------|
| TODO(validation) | High | Low | 1 | None |
| TODO(hints) | High | Medium | 2 | Validation |
| TODO(android-tv) | Medium | Medium | 3 | None |
| TODO(expansion) | Medium | Low | 4 | None |
| TODO(large-grids) | Medium | High | 5 | 64-bit clues |
| TODO(operations) | Low | High | 6 | Solver extensions |

---

## Synthesis: The Ultrafun Vision

### Core Loop Enhancement
1. **Instant Feedback**: Real-time validation highlights errors as you type
2. **Progressive Learning**: Hints teach techniques, not just answers
3. **Accessible Everywhere**: Touch, D-pad, keyboard all work flawlessly

### Advanced Features
4. **Research Mode**: GCD/LCM/MOD operations for math enthusiasts
5. **Massive Grids**: 16x16 supported with AI-assisted generation
6. **Speed Runs**: Timer mode with leaderboards

### Technical Excellence
7. **Zero-latency Generation**: AI + C hybrid for instant puzzles
8. **Offline-first**: All features work without network
9. **Cross-platform**: Phone, tablet, TV, Chromebook

---

## Next Steps

1. [ ] Add submodules for latin-square-toolbox and dancing-links
2. [ ] Implement TODO(validation) as foundation for hints
3. [ ] Port difficulty grading from billabob/kenkensolver
4. [ ] Create focus management infrastructure for TV
5. [ ] Train expanded 16x16 AI model
6. [ ] Design "Research Mode" UI for new operations

---

## Sources

- [Kenny - Rust KenKen Generator](https://github.com/camsteffen/kenny)
- [CanCan - Scala Solver/Generator](https://github.com/wpm/CanCan)
- [Latin Square Toolbox](https://github.com/nathanoschmidt/latin-square-toolbox)
- [SGT Puzzles Android](https://github.com/chrisboyle/sgtpuzzles)
- [KenKen Solver & Grader](https://billabob.github.io/kenkensolver/)
- [D-pad Compose Tutorial](https://github.com/thesauri/dpad-compose)
- [Android Focus in Compose](https://developer.android.com/develop/ui/compose/touch-input/focus)
- [Dancing Links C++](https://github.com/gkaranikas/dancing-links)
- [Latin Square CSP](https://github.com/3N4N/csp-latin-square)
- [Latin Square Completion](https://github.com/PritK99/Latin-Square-Completion)
