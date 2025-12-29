"""
ONNX Export Validation Tests

Tests P1.4.3:
- ONNX export produces valid model
- Output shape is [batch, 35, 16, 16] (35 classes)
- Input accepts VOCAB.EMPTY tokens
- Inference runs without errors

Run with: pytest test_onnx_export.py -v

SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: Copyright (C) 2025 KeenKenning Contributors
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from token_vocabulary import VOCAB, GameMode


class MinimalTransformer(nn.Module):
    """Minimal transformer for ONNX export testing."""

    def __init__(self, max_size=16, num_classes=35, d_model=64, n_layer=2):
        super().__init__()
        self.max_size = max_size
        self.num_classes = num_classes
        self.d_model = d_model

        self.tok_emb = nn.Embedding(num_classes, d_model)
        self.pos_emb = nn.Embedding(max_size * max_size, d_model)
        self.size_emb = nn.Embedding(max_size + 1, d_model)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=d_model * 4, batch_first=True)
            for _ in range(n_layer)
        ])

        self.head = nn.Linear(d_model, num_classes, bias=False)

    def forward(self, grid, size):
        B = grid.size(0)
        flat = grid.view(B, -1)
        seq_len = flat.size(1)

        tok = self.tok_emb(flat)
        pos = self.pos_emb(torch.arange(seq_len, device=grid.device).unsqueeze(0).expand(B, -1))
        sz = self.size_emb(size).unsqueeze(1).expand(B, seq_len, -1)

        h = tok + pos + sz
        for layer in self.layers:
            h = layer(h)

        logits = self.head(h)
        return logits.view(B, self.num_classes, self.max_size, self.max_size)


class TestONNXExport:
    """Test ONNX export with new 35-token vocabulary."""

    @pytest.fixture
    def model(self):
        """Create a minimal model for testing."""
        return MinimalTransformer(
            max_size=16,
            num_classes=VOCAB.VOCAB_SIZE,
            d_model=64,
            n_layer=2,
        )

    @pytest.fixture
    def temp_onnx_path(self):
        """Create a temporary path for ONNX file."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = Path(f.name)
        yield path
        if path.exists():
            path.unlink()

    def test_model_output_shape(self, model):
        """Model output should be [batch, 35, 16, 16]."""
        model.train(False)
        with torch.no_grad():
            grid = torch.full((2, 16, 16), VOCAB.EMPTY, dtype=torch.long)
            size = torch.tensor([6, 9], dtype=torch.long)
            output = model(grid, size)

        assert output.shape == (2, 35, 16, 16)

    def test_model_accepts_empty_tokens(self, model):
        """Model should accept EMPTY tokens as input."""
        model.train(False)
        with torch.no_grad():
            grid = torch.full((1, 16, 16), VOCAB.EMPTY, dtype=torch.long)
            size = torch.tensor([9], dtype=torch.long)
            output = model(grid, size)

        assert output.shape == (1, 35, 16, 16)
        assert not torch.isnan(output).any()

    def test_model_accepts_value_tokens(self, model):
        """Model should accept value tokens (PAD, EMPTY, digits)."""
        model.train(False)
        with torch.no_grad():
            grid = torch.zeros((1, 16, 16), dtype=torch.long)
            grid[0, 0, 0] = VOCAB.PAD
            grid[0, 0, 1] = VOCAB.EMPTY
            grid[0, 1, 0] = VOCAB.digit_to_token(1)
            grid[0, 1, 1] = VOCAB.digit_to_token(9)
            grid[0, 2, 0] = VOCAB.digit_to_token(0)
            grid[0, 2, 1] = VOCAB.digit_to_token(-3)

            size = torch.tensor([9], dtype=torch.long)
            output = model(grid, size)

        assert output.shape == (1, 35, 16, 16)
        assert not torch.isnan(output).any()

    def test_onnx_export_succeeds(self, model, temp_onnx_path):
        """ONNX export should succeed without errors."""
        model.train(False)

        dummy_grid = torch.full((1, 16, 16), VOCAB.EMPTY, dtype=torch.long)
        dummy_size = torch.tensor([9], dtype=torch.long)

        torch.onnx.export(
            model,
            (dummy_grid, dummy_size),
            str(temp_onnx_path),
            input_names=["input_grid", "grid_size"],
            output_names=["cell_logits"],
            dynamic_axes={
                "input_grid": {0: "batch"},
                "grid_size": {0: "batch"},
                "cell_logits": {0: "batch"},
            },
            opset_version=17,
        )

        assert temp_onnx_path.exists()
        assert temp_onnx_path.stat().st_size > 0


class TestVocabularyIntegration:
    """Test vocabulary integration in ONNX workflow."""

    def test_allowed_tokens_within_vocab_size(self):
        """All allowed tokens should be < VOCAB_SIZE."""
        for mode in [GameMode.STANDARD, GameMode.ZERO_INCLUSIVE, GameMode.NEGATIVE]:
            for size in [3, 4, 5, 6, 9, 12, 16]:
                tokens = VOCAB.get_allowed_tokens(mode, size)
                for t in tokens:
                    assert 0 <= t < VOCAB.VOCAB_SIZE

    def test_digit_tokens_for_argmax(self):
        """Verify token-to-digit mapping for argmax decoding."""
        tokens = VOCAB.get_allowed_tokens(GameMode.STANDARD, 9)
        assert len(tokens) == 9
        digits = [VOCAB.token_to_digit(t) for t in tokens]
        assert digits == [1, 2, 3, 4, 5, 6, 7, 8, 9]

        tokens = VOCAB.get_allowed_tokens(GameMode.ZERO_INCLUSIVE, 9)
        digits = [VOCAB.token_to_digit(t) for t in tokens]
        assert digits == [0, 1, 2, 3, 4, 5, 6, 7, 8]

        tokens = VOCAB.get_allowed_tokens(GameMode.NEGATIVE, 6)
        digits = sorted([VOCAB.token_to_digit(t) for t in tokens])
        assert digits == [-3, -2, -1, 1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
