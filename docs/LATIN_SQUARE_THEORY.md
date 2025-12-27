# Latin Square Theory & AI Synthesis
*As of December 24, 2025*

## 1. Theoretical Foundations

### 1.1 Combinatorial Structure
A **Latin Square** of order $N$ is an $N \times N$ array where each cell contains a symbol from {1, ..., N} such that each symbol appears exactly once in every row and column. This is a specific type of **Quasigroup** (algebraic structure with division).

For **KenKen (Keen)** puzzles, the Latin Square serves as the hidden "answer key" or "ground truth". The puzzle difficulty arises not just from the grid itself, but from the **Cage Constraints** applied on top of it.

### 1.2 Generation Algorithms
*   **Backtracking:** The exhaustive search method. Simple but exponential time complexity $O(N!)$. Efficient for small $N$ ($N \le 9$) but struggles with scale.
*   **Max-Flow / Bipartite Matching:**
    *   The problem of adding a valid row to a partial Latin Square of size $r \times N$ is equivalent to finding a **System of Distinct Representatives (SDR)** or a perfect matching in a bipartite graph.
    *   **Hall's Marriage Theorem** guarantees that if a partial Latin Square can be extended, a matching exists.
    *   Our C-backend (`maxflow.c`) utilizes this property (Min-Cost Max-Flow) to efficiently generate valid rows, avoiding the "dead ends" typical of naive backtracking. This makes it an ideal "Teacher" for generating high-quality training data.

## 2. Machine Learning Approaches (2024-2025 State of the Art)

### 2.1 Why CNNs Struggle
Standard Convolutional Neural Networks (CNNs) rely on *local* spatial invariance. However, Latin Square constraints are **global** and **rigid**. A change in cell (0,0) restricts the possibilities for (0, N-1) and (N-1, 0) instantly. Pure CNNs often fail to capture these long-range dependencies without immense depth or fully connected layers.

### 2.2 Effective Architectures
*   **Recurrent Relational Networks (RRNs):** Explicitly model the graph structure of the puzzle (cells are nodes, constraints are edges). Achieves >96% accuracy on hard instances.
*   **Transformer / Attention Models:** Self-attention mechanisms ($Attention(Q, K, V)$) naturally capture global dependencies ($N \times N$ range) better than convolution kernels. Treating the grid as a sequence of tokens allows the model to "attend" to the relevant row/column peers.
*   **Physics-Inspired (Oscillatory/Diffusion):** While powerful, these are currently too computationally heavy for on-device Android inference via ONNX Runtime.

## 3. "Keen" Specific Difficulty Theory
A "Hard" Keen puzzle is defined by:
1.  **Minimization of "Freebies":** Few or no single-cell cages.
2.  **Cage Interdependence:** Cages where the candidate set depends on the resolution of a neighbor.
3.  **Arithmetic Ambiguity:** Cages like "Target 12, Multiply" (3x4 vs 2x6) or "Target 1, Subtract" (many pairs).

## 4. Synthesis for Orthogon

To maximize the game's potential, we will adopt a **Teacher-Student Architecture**:

1.  **Teacher (The C Engine):**
    *   Uses rigorous `maxflow` algorithms to generate *guaranteed valid* Latin Squares.
    *   Can run "soak" tests to produce massive, unbiased datasets ($10^5$ grids).

2.  **Student (The Python AI):**
    *   **Goal:** Learn the underlying distribution of valid Latin Squares.
    *   **Model:** A **Small ResNet** or **Dilated CNN**. We choose this over Transformers for *speed* and *size* ("Tiny" constraint) on mobile. Dilated convolutions allow the receptive field to expand to the full grid width/height without excessive parameters.
    *   **Input:** An empty or partially filled grid (if we expand to completion tasks).
    *   **Output:** A valid full grid.

3.  **The Feedback Loop:**
    *   The AI generates a grid.
    *   The C Engine *validates* it and then *generates clues* around it.
    *   This hybrid approach allows the AI to act as a fast "Idea Generator" (providing the Latin Square) while the C Engine acts as the "Dungeon Master" (building the cage walls).

## 5. Execution Roadmap
1.  **Data Gen:** `generate_data.py` wrapping `latin_gen`.
2.  **Training:** `train_real_model.py` with a custom ResNet/CNN.
3.  **Inference:** Export to ONNX, integrate into Android.
4.  **UI:** Update game to show "Neural" vs "Classic" generation stats.
