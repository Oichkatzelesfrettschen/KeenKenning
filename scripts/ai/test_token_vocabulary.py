"""
Unit Tests for Token Vocabulary

Tests P1.5.1-P1.5.2:
- Round-trip digit <-> token conversion
- ZERO mode token mapping
- NEGATIVE mode token mapping
- Edge cases and error handling

Run with: pytest test_token_vocabulary.py -v

SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: Copyright (C) 2025 KeenKenning Contributors
"""

import pytest

from token_vocabulary import (
    GameMode,
    TokenVocabulary,
    VOCAB,
    digit_to_token,
    token_to_digit,
    get_allowed_tokens,
)


class TestTokenVocabularyConstants:
    """Test vocabulary constants and structure."""

    def test_vocab_size(self):
        """Vocabulary should have exactly 35 tokens."""
        assert VOCAB.VOCAB_SIZE == 35

    def test_pad_token(self):
        """PAD should be token 0."""
        assert VOCAB.PAD == 0

    def test_empty_token(self):
        """EMPTY should be token 1."""
        assert VOCAB.EMPTY == 1

    def test_val_offset_positive(self):
        """Positive digit offset should be 2."""
        assert VOCAB.VAL_OFFSET_POSITIVE == 2

    def test_val_offset_negative(self):
        """Negative digit offset should be 35."""
        assert VOCAB.VAL_OFFSET_NEGATIVE == 35


class TestDigitToTokenRoundTrip:
    """Test bidirectional digit <-> token conversion."""

    @pytest.mark.parametrize("digit", range(-16, 17))
    def test_roundtrip_all_digits(self, digit):
        """All digits from -16 to 16 should round-trip correctly."""
        token = digit_to_token(digit)
        result = token_to_digit(token)
        assert result == digit, f"digit {digit} -> token {token} -> {result}"

    def test_digit_zero_is_not_pad(self):
        """Digit 0 should NOT map to PAD token."""
        token = digit_to_token(0)
        assert token != VOCAB.PAD
        assert token == 2  # VAL_0

    def test_digit_zero_is_not_empty(self):
        """Digit 0 should NOT map to EMPTY token."""
        token = digit_to_token(0)
        assert token != VOCAB.EMPTY

    def test_positive_digit_range(self):
        """Positive digits 0-16 should map to tokens 2-18."""
        for digit in range(17):
            token = digit_to_token(digit)
            assert 2 <= token <= 18

    def test_negative_digit_range(self):
        """Negative digits -1 to -16 should map to tokens 19-34."""
        for digit in range(-16, 0):
            token = digit_to_token(digit)
            assert 19 <= token <= 34


class TestTokenToDigitErrors:
    """Test error handling for invalid tokens."""

    def test_pad_token_raises(self):
        """Converting PAD token should raise ValueError."""
        with pytest.raises(ValueError, match="PAD"):
            token_to_digit(VOCAB.PAD)

    def test_empty_token_raises(self):
        """Converting EMPTY token should raise ValueError."""
        with pytest.raises(ValueError, match="EMPTY"):
            token_to_digit(VOCAB.EMPTY)

    def test_negative_token_raises(self):
        """Negative tokens should raise ValueError."""
        with pytest.raises(ValueError):
            token_to_digit(-1)

    def test_out_of_range_token_raises(self):
        """Tokens >= VOCAB_SIZE should raise ValueError."""
        with pytest.raises(ValueError):
            token_to_digit(35)
        with pytest.raises(ValueError):
            token_to_digit(100)


class TestDigitToTokenErrors:
    """Test error handling for invalid digits."""

    def test_digit_too_large_raises(self):
        """Digit > 16 should raise ValueError."""
        with pytest.raises(ValueError, match="exceeds max"):
            digit_to_token(17)

    def test_digit_too_small_raises(self):
        """Digit < -16 should raise ValueError."""
        with pytest.raises(ValueError, match="exceeds min"):
            digit_to_token(-17)


class TestIsValueToken:
    """Test is_value_token() helper."""

    def test_pad_is_not_value(self):
        """PAD should not be a value token."""
        assert not VOCAB.is_value_token(VOCAB.PAD)

    def test_empty_is_not_value(self):
        """EMPTY should not be a value token."""
        assert not VOCAB.is_value_token(VOCAB.EMPTY)

    @pytest.mark.parametrize("token", range(2, 35))
    def test_all_value_tokens(self, token):
        """Tokens 2-34 should all be value tokens."""
        assert VOCAB.is_value_token(token)


class TestStandardModeTokens:
    """Test allowed tokens for STANDARD mode (digits 1-N)."""

    @pytest.mark.parametrize("size", [3, 4, 5, 6, 9, 12, 16])
    def test_standard_mode_digits(self, size):
        """STANDARD mode should have digits 1 to N."""
        tokens = get_allowed_tokens(GameMode.STANDARD, size)
        digits = [token_to_digit(t) for t in tokens]
        expected = list(range(1, size + 1))
        assert digits == expected

    def test_standard_mode_excludes_zero(self):
        """STANDARD mode should not include digit 0."""
        tokens = get_allowed_tokens(GameMode.STANDARD, 6)
        digits = [token_to_digit(t) for t in tokens]
        assert 0 not in digits


class TestZeroInclusiveModeTokens:
    """Test allowed tokens for ZERO_INCLUSIVE mode (digits 0 to N-1)."""

    @pytest.mark.parametrize("size", [3, 4, 5, 6, 9, 12, 16])
    def test_zero_mode_digits(self, size):
        """ZERO mode should have digits 0 to N-1."""
        tokens = get_allowed_tokens(GameMode.ZERO_INCLUSIVE, size)
        digits = [token_to_digit(t) for t in tokens]
        expected = list(range(size))
        assert digits == expected

    def test_zero_mode_includes_zero(self):
        """ZERO mode MUST include digit 0."""
        tokens = get_allowed_tokens(GameMode.ZERO_INCLUSIVE, 6)
        digits = [token_to_digit(t) for t in tokens]
        assert 0 in digits

    def test_zero_mode_token_for_zero(self):
        """In ZERO mode, digit 0 should have its own unique token."""
        tokens = get_allowed_tokens(GameMode.ZERO_INCLUSIVE, 6)
        # Find the token for digit 0
        zero_token = digit_to_token(0)
        assert zero_token in tokens
        assert zero_token != VOCAB.PAD
        assert zero_token != VOCAB.EMPTY


class TestNegativeModeTokens:
    """Test allowed tokens for NEGATIVE mode (symmetric around 0)."""

    def test_negative_mode_odd_size_includes_zero(self):
        """Odd sizes should include 0."""
        for size in [3, 5, 7, 9]:
            tokens = get_allowed_tokens(GameMode.NEGATIVE, size)
            digits = [token_to_digit(t) for t in tokens]
            assert 0 in digits, f"size={size} should include 0"

    def test_negative_mode_even_size_excludes_zero(self):
        """Even sizes should exclude 0."""
        for size in [4, 6, 8, 10]:
            tokens = get_allowed_tokens(GameMode.NEGATIVE, size)
            digits = [token_to_digit(t) for t in tokens]
            assert 0 not in digits, f"size={size} should exclude 0"

    def test_negative_mode_symmetric(self):
        """Digits should be symmetric around 0."""
        for size in [3, 4, 5, 6, 7, 8]:
            tokens = get_allowed_tokens(GameMode.NEGATIVE, size)
            digits = set(token_to_digit(t) for t in tokens)
            # For each positive digit, there should be a corresponding negative
            for d in digits:
                if d != 0:
                    assert -d in digits, f"size={size}: {d} without {-d}"

    def test_negative_mode_size_3(self):
        """Size 3 NEGATIVE: {-1, 0, 1}."""
        tokens = get_allowed_tokens(GameMode.NEGATIVE, 3)
        digits = sorted([token_to_digit(t) for t in tokens])
        assert digits == [-1, 0, 1]

    def test_negative_mode_size_4(self):
        """Size 4 NEGATIVE: {-2, -1, 1, 2}."""
        tokens = get_allowed_tokens(GameMode.NEGATIVE, 4)
        digits = sorted([token_to_digit(t) for t in tokens])
        assert digits == [-2, -1, 1, 2]

    def test_negative_mode_size_6(self):
        """Size 6 NEGATIVE: {-3, -2, -1, 1, 2, 3}."""
        tokens = get_allowed_tokens(GameMode.NEGATIVE, 6)
        digits = sorted([token_to_digit(t) for t in tokens])
        assert digits == [-3, -2, -1, 1, 2, 3]

    def test_negative_mode_has_correct_count(self):
        """NEGATIVE mode should have exactly N tokens."""
        for size in [3, 4, 5, 6, 7, 8, 9]:
            tokens = get_allowed_tokens(GameMode.NEGATIVE, size)
            assert len(tokens) == size, f"size={size} should have {size} tokens"


class TestAllowedMask:
    """Test create_allowed_mask() helper."""

    def test_mask_length(self):
        """Mask should have VOCAB_SIZE elements."""
        mask = VOCAB.create_allowed_mask(GameMode.STANDARD, 6)
        assert len(mask) == VOCAB.VOCAB_SIZE

    def test_mask_pad_always_false(self):
        """PAD should never be allowed."""
        for mode in [GameMode.STANDARD, GameMode.ZERO_INCLUSIVE, GameMode.NEGATIVE]:
            mask = VOCAB.create_allowed_mask(mode, 6)
            assert not mask[VOCAB.PAD]

    def test_mask_empty_always_false(self):
        """EMPTY should never be in allowed mask (it's input, not output)."""
        for mode in [GameMode.STANDARD, GameMode.ZERO_INCLUSIVE, GameMode.NEGATIVE]:
            mask = VOCAB.create_allowed_mask(mode, 6)
            assert not mask[VOCAB.EMPTY]

    def test_mask_matches_allowed_tokens(self):
        """Mask should match get_allowed_tokens() results."""
        for mode in [GameMode.STANDARD, GameMode.ZERO_INCLUSIVE, GameMode.NEGATIVE]:
            for size in [4, 6, 9]:
                mask = VOCAB.create_allowed_mask(mode, size)
                tokens = get_allowed_tokens(mode, size)
                for t in range(VOCAB.VOCAB_SIZE):
                    assert mask[t] == (t in tokens)


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_digit_to_token_uses_singleton(self):
        """Module function should use VOCAB singleton."""
        assert digit_to_token(5) == VOCAB.digit_to_token(5)

    def test_token_to_digit_uses_singleton(self):
        """Module function should use VOCAB singleton."""
        assert token_to_digit(7) == VOCAB.token_to_digit(7)

    def test_get_allowed_tokens_uses_singleton(self):
        """Module function should use VOCAB singleton."""
        result = get_allowed_tokens(GameMode.STANDARD, 6)
        expected = VOCAB.get_allowed_tokens(GameMode.STANDARD, 6)
        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
