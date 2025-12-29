"""
Token Vocabulary for KeenKenning ML Training Pipeline

Defines the unified token vocabulary that fixes the token 0 dual-use bug.
Each semantic concept has a unique token ID.

Token Layout (35 tokens):
  [0]     PAD      - Padding for cells outside grid bounds
  [1]     EMPTY    - Masked/unknown cells during autoregressive training
  [2-18]  VAL_0-16 - Positive digits (0 to 16)
  [19-34] VAL_-1   - Negative digits (-1 to -16)

This eliminates ambiguity where token 0 previously meant both
"padding" and "digit 0 in ZERO mode".

SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: Copyright (C) 2025 KeenKenning Contributors
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Set, Tuple


class GameMode(IntEnum):
    """Game mode flags matching keen_modes.h and GameMode.kt.

    Values match the C header MODE_* flags for consistency.
    """

    STANDARD = 0
    ZERO_INCLUSIVE = 1  # 0 to N-1 instead of 1 to N
    MULTIPLICATIVE = 2  # Only * and / operations
    KILLER = 3  # No duplicate digits in cages
    MODULAR = 4  # Modular arithmetic for clues
    EXPONENT = 5  # Exponent operations
    NEGATIVE = 6  # Symmetric negative/positive digits
    ROMAN = 7  # Display as Roman numerals
    HEX = 8  # Display as hexadecimal
    CLASSIK = 9  # Classic mode (no ops, sums only)
    RETRO_8BIT = 10  # Visual theme only


@dataclass(frozen=True)
class TokenVocabulary:
    """Unified token vocabulary for Latin square generation.

    Frozen dataclass ensures vocabulary is immutable after creation.
    """

    # Special tokens (semantic roles)
    PAD: int = 0  # Padding for out-of-bounds cells
    EMPTY: int = 1  # Unknown/masked cells

    # Value token offset: token = VAL_OFFSET + digit
    # For digit 0: token = 2
    # For digit 16: token = 18
    # For digit -1: token = 19
    # For digit -16: token = 34
    VAL_OFFSET_POSITIVE: int = 2  # digit d -> token = 2 + d
    VAL_OFFSET_NEGATIVE: int = 35  # digit -d -> token = 35 - d (so -1 -> 34, -16 -> 19)

    # Total vocabulary size
    VOCAB_SIZE: int = 35

    # Value ranges
    MAX_POSITIVE_DIGIT: int = 16
    MAX_NEGATIVE_DIGIT: int = -16

    def digit_to_token(self, digit: int) -> int:
        """Convert a grid digit value to its token ID.

        Args:
            digit: A grid value (-16 to 16, excluding ambiguous cases per mode)

        Returns:
            Token ID in range [2, 34]

        Raises:
            ValueError: If digit is out of valid range
        """
        if digit >= 0:
            if digit > self.MAX_POSITIVE_DIGIT:
                raise ValueError(f"Positive digit {digit} exceeds max {self.MAX_POSITIVE_DIGIT}")
            return self.VAL_OFFSET_POSITIVE + digit
        else:
            if digit < self.MAX_NEGATIVE_DIGIT:
                raise ValueError(f"Negative digit {digit} exceeds min {self.MAX_NEGATIVE_DIGIT}")
            # -1 -> 34, -2 -> 33, ..., -16 -> 19
            return self.VAL_OFFSET_NEGATIVE + digit  # 35 + (-1) = 34

    def token_to_digit(self, token: int) -> int:
        """Convert a token ID back to its grid digit value.

        Args:
            token: Token ID in range [2, 34]

        Returns:
            Grid digit value (-16 to 16)

        Raises:
            ValueError: If token is PAD, EMPTY, or out of range
        """
        if token == self.PAD:
            raise ValueError("Cannot convert PAD token to digit")
        if token == self.EMPTY:
            raise ValueError("Cannot convert EMPTY token to digit")
        if token < self.VAL_OFFSET_POSITIVE or token >= self.VOCAB_SIZE:
            raise ValueError(f"Token {token} is out of valid range [2, {self.VOCAB_SIZE - 1}]")

        # Tokens 2-18 are positive digits 0-16
        if token <= self.VAL_OFFSET_POSITIVE + self.MAX_POSITIVE_DIGIT:
            return token - self.VAL_OFFSET_POSITIVE
        else:
            # Tokens 19-34 are negative digits -16 to -1
            return token - self.VAL_OFFSET_NEGATIVE  # 34 - 35 = -1

    def is_value_token(self, token: int) -> bool:
        """Check if a token represents a grid value (not PAD or EMPTY)."""
        return self.VAL_OFFSET_POSITIVE <= token < self.VOCAB_SIZE

    def get_allowed_tokens(self, mode: GameMode, size: int) -> List[int]:
        """Get the set of valid value tokens for a given mode and grid size.

        Args:
            mode: The game mode determining digit range
            size: Grid size (N)

        Returns:
            List of valid token IDs for this mode/size combination
        """
        if mode == GameMode.ZERO_INCLUSIVE:
            # Digits 0 to N-1
            return [self.digit_to_token(d) for d in range(size)]

        elif mode == GameMode.NEGATIVE:
            # Symmetric digits centered at 0
            # For odd N: include 0, e.g., N=5 -> {-2,-1,0,1,2}
            # For even N: exclude 0, e.g., N=6 -> {-3,-2,-1,1,2,3}
            half = size // 2
            if size % 2 == 1:
                # Odd: -half to +half inclusive
                digits = list(range(-half, half + 1))
            else:
                # Even: -half to -1, then 1 to +half (exclude 0)
                digits = list(range(-half, 0)) + list(range(1, half + 1))
            return [self.digit_to_token(d) for d in digits]

        else:
            # STANDARD and most other modes: digits 1 to N
            return [self.digit_to_token(d) for d in range(1, size + 1)]

    def get_digit_range(self, mode: GameMode, size: int) -> Tuple[int, int]:
        """Get the min and max digit values for a mode/size combination.

        Returns:
            (min_digit, max_digit) tuple
        """
        if mode == GameMode.ZERO_INCLUSIVE:
            return (0, size - 1)
        elif mode == GameMode.NEGATIVE:
            half = size // 2
            if size % 2 == 1:
                return (-half, half)
            else:
                return (-half, half)  # Even though 0 is excluded
        else:
            return (1, size)

    def create_allowed_mask(self, mode: GameMode, size: int) -> List[bool]:
        """Create a boolean mask for allowed tokens.

        Args:
            mode: Game mode
            size: Grid size

        Returns:
            List of VOCAB_SIZE booleans, True for allowed tokens
        """
        allowed_set = set(self.get_allowed_tokens(mode, size))
        return [i in allowed_set for i in range(self.VOCAB_SIZE)]


# Singleton instance for convenience
VOCAB = TokenVocabulary()


# Convenience functions that use the singleton
def digit_to_token(digit: int) -> int:
    """Convert digit to token using default vocabulary."""
    return VOCAB.digit_to_token(digit)


def token_to_digit(token: int) -> int:
    """Convert token to digit using default vocabulary."""
    return VOCAB.token_to_digit(token)


def get_allowed_tokens(mode: GameMode, size: int) -> List[int]:
    """Get allowed tokens for mode/size using default vocabulary."""
    return VOCAB.get_allowed_tokens(mode, size)


if __name__ == "__main__":
    # Quick sanity check
    print("=== Token Vocabulary Sanity Check ===\n")

    v = TokenVocabulary()

    print(f"Vocabulary size: {v.VOCAB_SIZE}")
    print(f"PAD token: {v.PAD}")
    print(f"EMPTY token: {v.EMPTY}")
    print()

    # Test round-trip for all valid digits
    print("Round-trip test:")
    for digit in range(-16, 17):
        token = v.digit_to_token(digit)
        back = v.token_to_digit(token)
        status = "PASS" if back == digit else f"FAIL (got {back})"
        print(f"  {digit:3d} -> token {token:2d} -> {back:3d} : {status}")

    print()

    # Test allowed tokens for each mode
    print("Allowed tokens by mode (size=6):")
    for mode in [GameMode.STANDARD, GameMode.ZERO_INCLUSIVE, GameMode.NEGATIVE]:
        tokens = v.get_allowed_tokens(mode, 6)
        digits = [v.token_to_digit(t) for t in tokens]
        print(f"  {mode.name:15s}: digits {digits} -> tokens {tokens}")

    print()

    # Test NEGATIVE mode for different sizes
    print("NEGATIVE mode digit ranges:")
    for size in [3, 4, 5, 6, 7, 8]:
        tokens = v.get_allowed_tokens(GameMode.NEGATIVE, size)
        digits = [v.token_to_digit(t) for t in tokens]
        print(f"  size={size}: {digits}")
