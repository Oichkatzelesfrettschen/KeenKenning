# Game Modes Architecture Design

## Overview

This document outlines the architecture for implementing additional KenKen game
modes in KeenKeen for Android.

## Mode Complexity Analysis

| Mode               | C Changes    | Kotlin/UI    | Solver Impact | Priority |
|-------------------|--------------|--------------|---------------|----------|
| No-Op/Mystery     | None         | UI display   | None          | P1       |
| Zero-Inclusive    | Range param  | Minor        | Low           | P1       |
| 12x12/16x16       | Already OK   | Hex digits   | None          | P1       |
| Exponentiation    | New C_POW    | Minor        | Medium        | P2       |
| Negative Numbers  | Signed math  | Signed UI    | Medium        | P2       |
| Modular Arith     | Mod operator | UI indicator | High          | P3       |
| Killer Cages      | Constraints  | Visual cues  | High          | P3       |

## Architecture Design

### 1. GameMode Enum (Kotlin)

```kotlin
enum class GameMode(
    val displayName: String,
    val description: String,
    val icon: ImageVector,
    val cFlags: Int  // Passed to JNI
) {
    STANDARD(
        "Standard",
        "All operations (+, -, ×, ÷)",
        Icons.Default.Calculate,
        0x00
    ),
    MULTIPLICATION_ONLY(
        "Multiplication",
        "Only × operations",
        Icons.Default.Close,
        0x01
    ),
    MYSTERY(
        "Mystery",
        "Operations hidden - deduce them!",
        Icons.Default.Help,
        0x02
    ),
    ZERO_INCLUSIVE(
        "Zero Mode",
        "Numbers 0 to N-1",
        Icons.Default.Exposure,
        0x04
    ),
    NEGATIVE_NUMBERS(
        "Negative",
        "Range -N to +N",
        Icons.Default.Remove,
        0x08
    ),
    EXPONENT(
        "Powers",
        "Includes ^ exponent operation",
        Icons.Default.Superscript,
        0x10
    ),
    MODULAR(
        "Modular",
        "Wrap-around arithmetic (mod N)",
        Icons.Default.Loop,
        0x20
    ),
    KILLER(
        "Killer",
        "No repeated digits in cages",
        Icons.Default.Block,
        0x40
    )
}
```

### 2. C-Layer Changes (keen.h)

```c
struct game_params {
    int w;                    // Grid width (3-16)
    int diff;                 // Difficulty level
    int mode_flags;           // Bit flags for game mode
    // Flags:
    // 0x01 = multiplication_only
    // 0x02 = mystery_mode (hide operations)
    // 0x04 = zero_inclusive
    // 0x08 = negative_numbers
    // 0x10 = exponent_enabled
    // 0x20 = modular_arithmetic
    // 0x40 = killer_cages
};

// New operation constant
#define C_POW 0x80000000L
```

### 3. Menu UI Design - Horizontal Scrolling Cards

```
┌──────────────────────────────────────────────────────────────┐
│ GAME MODE                                                    │
│ ◀ [Standard] [Multiply] [Mystery] [Zero] [Negative] ... ▶    │
│     ━━━━━━━                                                  │
│   "All operations (+, -, ×, ÷)"                              │
└──────────────────────────────────────────────────────────────┘
```

**Why scrolling cards over dropdown:**
1. Visual preview of all modes without extra tap
2. Touch-friendly on mobile
3. Can show mode description inline
4. Easy to add new modes later
5. More engaging UX than dropdown

### 4. Implementation Phases

#### Phase 1: Low-Effort Modes (1-2 days each)

**1a. Mystery/No-Op Mode**
- Changes: UI only - hide operation symbol in cage display
- Files: `GameScreen.kt`, `GameUiState.kt`
- No solver changes needed

**1b. Zero-Inclusive Mode**
- Changes: Shift number range from [1,N] to [0,N-1]
- Files: `keen.c` (latin square gen), `GameScreen.kt` (display)
- Solver already handles any digit values

**1c. Extended Grid Sizes (10-16)**
- Already supported in C layer (`w` parameter)
- Changes: `MenuScreen.kt` (extend size selector)
- Add hex digit display (A-G) for values > 9

#### Phase 2: Medium-Effort Modes (3-5 days each)

**2a. Exponentiation Mode**
- Add `C_POW` constant in `keen.c`
- Add power operation handling in solver
- Update clue generation to include `^`
- Files: `keen.c`, `GameScreen.kt`

**2b. Negative Numbers Mode**
- Range becomes [-N, N] excluding 0
- Requires signed arithmetic in solver
- UI needs to handle minus signs in cells
- Files: `keen.c`, `latin.c`, UI files

#### Phase 3: High-Effort Modes (1-2 weeks each)

**3a. Modular Arithmetic**
- All operations computed mod N
- New solver logic for wrap-around
- May affect difficulty balancing
- Files: `keen.c` (extensive changes)

**3b. Killer Cages**
- Add uniqueness constraint within cages
- New solver rule: no digit repeats in cage
- Similar to Killer Sudoku constraint
- Files: `keen.c` (solver), new constraint system

### 5. Data Flow

```
MenuScreen.kt
    │
    ▼ selectedMode: GameMode
MenuActivity.kt
    │
    ▼ mode.cFlags
KeenModelBuilder.java (JNI)
    │
    ▼ modeFlags param
keen-android-jni.c
    │
    ▼ params.mode_flags
keen.c (solver/generator)
```

### 6. MenuScreen.kt Modifications

Replace toggle options with scrollable mode selector:

```kotlin
@Composable
fun GameModeSelector(
    selectedMode: GameMode,
    onModeChange: (GameMode) -> Unit
) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Text(
            text = "Game Mode",
            fontSize = 12.sp,
            color = Color(0xFF888888)
        )

        LazyRow(
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            contentPadding = PaddingValues(vertical = 8.dp)
        ) {
            items(GameMode.values()) { mode ->
                ModeCard(
                    mode = mode,
                    isSelected = mode == selectedMode,
                    onClick = { onModeChange(mode) }
                )
            }
        }

        // Description of selected mode
        Text(
            text = selectedMode.description,
            fontSize = 14.sp,
            color = Color(0xFFAAAAAA),
            modifier = Modifier.padding(top = 4.dp)
        )
    }
}

@Composable
fun ModeCard(
    mode: GameMode,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .width(100.dp)
            .clickable(onClick = onClick),
        colors = CardDefaults.cardColors(
            containerColor = if (isSelected)
                Color(0xFF7B68EE)
            else
                Color(0xFF2a2a4a)
        ),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = mode.icon,
                contentDescription = mode.displayName,
                tint = if (isSelected) Color.White else Color(0xFF888888)
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = mode.displayName,
                fontSize = 12.sp,
                color = if (isSelected) Color.White else Color(0xFFCCCCCC),
                maxLines = 1
            )
        }
    }
}
```

### 7. Backward Compatibility

- Default mode = STANDARD (matches current behavior)
- Saved games store mode flag
- Old saves without mode flag treated as STANDARD

### 8. Testing Strategy

Each mode needs:
1. Unit tests for solver correctness
2. UI tests for display
3. Puzzle generation verification
4. Save/load with mode persistence
5. Difficulty balancing per mode

## Recommendation

**Start with Phase 1** (Mystery, Zero-Inclusive, Extended Grids) as these:
- Have lowest implementation risk
- Provide immediate value to users
- Validate the mode infrastructure
- Can be shipped incrementally

The scrolling card selector should be implemented first to establish the
UI pattern before adding modes.

---

## Phase 4: Research-Backed Innovations

These novel modes synthesize insights from recent constraint satisfaction
research, competitive puzzle design, and unique gameplay innovations.

### 4.1 Explainable Hints Mode (HINT_MODE)

**Research Basis:** [Step-wise CSP Explanation](https://www.sciencedirect.com/science/article/pii/S0004370221001016)

**Concept:** Instead of revealing answers, the AI explains *why* a cell
must have a certain value using natural language:
- "Cell (2,3) must be 4 because it's the only value that satisfies the 12× cage"
- "Row 1 already has 1,2,3,5 so (1,4) must be 4"

**Implementation:**
- Add hint inference engine that tracks deduction chain
- UI displays reasoning steps
- Progressive hints: first vague, then specific

**Complexity:** MEDIUM
**Novel Factor:** HIGH (no KenKen app does explanatory hints)

### 4.2 Adaptive Difficulty Mode (ADAPTIVE)

**Research Basis:** [RL for Constraint Games](https://arxiv.org/abs/2102.06019)

**Concept:** Puzzle difficulty adjusts based on player performance:
- Track solve time, mistakes, hint usage
- Next puzzle calibrates to challenge level
- "Flow state" optimization - not too easy, not too hard

**Implementation:**
- Player profile with performance metrics
- Difficulty prediction model (simple heuristics or ML)
- Dynamic puzzle generation parameters

**Complexity:** MEDIUM-HIGH
**Novel Factor:** HIGH (personalized difficulty is rare)

### 4.3 Cascade Puzzles Mode (CASCADE)

**Concept:** Solving one cage affects constraints in adjacent cages:
- Cage solutions "unlock" adjacent cages
- Creates wave-like solving pattern
- Strategic order matters

**Example:**
```
┌─────┬─────┐
│ 6+  │ ?   │ <- "?" cage's target is revealed
│  3  │  ?  │    when 6+ cage is solved
├─────┼─────┤
│ ?   │ 12× │
│  ?  │  4  │
└─────┴─────┘
```

**Implementation:**
- Dependency graph between cages
- Progressive revelation system
- New UI for "locked" vs "unlocked" cages

**Complexity:** HIGH
**Novel Factor:** VERY HIGH (completely new mechanic)

### 4.4 Dueling KenKen (VERSUS)

**Concept:** Two players solve the same puzzle simultaneously:
- Race mode: first to complete wins
- Territory mode: claim cells, most territory wins
- Sabotage mode: can "lock" one opponent cell per minute

**Implementation:**
- Real-time sync (Firebase/WebSocket)
- Split-screen or overlay UI
- Matchmaking system

**Complexity:** VERY HIGH
**Novel Factor:** HIGH (competitive KenKen is novel)

### 4.5 Constraint Chain Mode (CHAIN)

**Research Basis:** [Transformer CSP Solving](https://arxiv.org/html/2307.04895)

**Concept:** Each cage has a secondary constraint that depends on
adjacent cage solutions:
- "This cage's sum = neighbor's product mod 5"
- Creates interdependent solving loops
- Requires holistic reasoning

**Implementation:**
- Extended constraint language
- Solver modifications for dependent constraints
- UI for showing constraint relationships

**Complexity:** VERY HIGH
**Novel Factor:** VERY HIGH (novel constraint type)

### 4.6 Story Puzzles Mode (NARRATIVE)

**Concept:** Puzzles with thematic constraints and narrative context:
- "Budget Mode": Balance income (positive) vs expenses (negative)
- "Recipe Mode": Ingredient quantities that sum to portions
- "Time Mode": Schedule constraints that don't overlap

**Implementation:**
- Thematic wrapper around existing mechanics
- Custom iconography per theme
- Narrative text between puzzles

**Complexity:** LOW-MEDIUM
**Novel Factor:** MEDIUM (thematic puzzles exist but are rare)

### 4.7 Meta-Grid Mode (META)

**Concept:** Solution digits form a pattern when viewed as a whole:
- Complete puzzle reveals a number/shape in the grid
- Secondary objective beyond just solving
- Aesthetic constraint satisfaction

**Example:**
```
Final solution reveals "7" pattern:
┌───┬───┬───┬───┐
│ 1 │ 1 │ 1 │ 1 │  <- row of same digits
├───┼───┼───┼───┤
│ 2 │ 3 │ 4 │ 1 │
├───┼───┼───┼───┤
│ 3 │ 4 │ 2 │ 1 │
├───┼───┼───┼───┤
│ 4 │ 2 │ 3 │ 1 │  <- diagonal pattern
└───┴───┴───┴───┘
```

**Implementation:**
- Puzzle generation with meta-constraints
- Pattern recognition display
- Achievement system for finding patterns

**Complexity:** MEDIUM
**Novel Factor:** HIGH (unique aesthetic layer)

### Phase 4 Priority Matrix

| Mode | Implementation | Novelty | User Appeal | Priority |
|------|---------------|---------|-------------|----------|
| Explainable Hints | MEDIUM | HIGH | VERY HIGH | P4.1 |
| Story Puzzles | LOW | MEDIUM | HIGH | P4.2 |
| Adaptive Difficulty | MEDIUM | HIGH | HIGH | P4.3 |
| Meta-Grid | MEDIUM | HIGH | MEDIUM | P4.4 |
| Cascade | HIGH | VERY HIGH | MEDIUM | P4.5 |
| Dueling | VERY HIGH | HIGH | HIGH | P4.6 |
| Constraint Chain | VERY HIGH | VERY HIGH | LOW | P4.7 |

### Sanity Check Matrix

Each Phase 4 mode validated against:

| Mode | Uses Existing Solver? | C Changes? | Networking? | Feasible? |
|------|----------------------|------------|-------------|-----------|
| Explainable Hints | YES (wraps) | Minor | NO | ✓ YES |
| Story Puzzles | YES | NO | NO | ✓ YES |
| Adaptive | YES | NO | NO | ✓ YES |
| Meta-Grid | YES (+ constraint) | Minor | NO | ✓ YES |
| Cascade | Partial | Moderate | NO | ✓ MAYBE |
| Dueling | YES | NO | YES | ⚠ COMPLEX |
| Constraint Chain | NO (new solver) | Major | NO | ⚠ RISKY |

### Recommended Phase 4 Scope

**Implement (Low Risk, High Value):**
1. Explainable Hints - Differentiated feature, uses existing solver
2. Story Puzzles - Thematic wrapper, no solver changes
3. Adaptive Difficulty - Profile-based, gradual rollout

**Defer (High Risk, Research Needed):**
- Cascade, Dueling, Constraint Chain require significant architecture work

---

## Sources

- [KenKen International Championship](https://www.kenkenchampionship.com/)
- [World Puzzle Championship 2024](https://wpc.puzzles.com/uspc2024/)
- [Deep Learning for CSPs (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/18011)
- [Recurrent Transformers for Puzzles](https://arxiv.org/html/2307.04895)
- [RL for Constraint Games](https://arxiv.org/abs/2102.06019)
- [Step-wise CSP Explanation](https://www.sciencedirect.com/science/article/pii/S0004370221001016)
