#!/usr/bin/env python3
"""
train_narrative_model.py: Train CharLSTM for puzzle narrative generation

Architecture:
- Character-level LSTM (no tokenizer needed)
- 2-layer bidirectional LSTM
- Target size: ~2MB (suitable for mobile)

SPDX-License-Identifier: MIT
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Lazy imports for torch
torch = None
nn = None
optim = None


def lazy_import_torch():
    """Import PyTorch lazily to allow script inspection without GPU."""
    global torch, nn, optim
    if torch is None:
        import torch as _torch
        import torch.nn as _nn
        import torch.optim as _optim
        torch = _torch
        nn = _nn
        optim = _optim


# =============================================================================
# HYPERPARAMETERS
# =============================================================================

EMBED_DIM = 32          # Character embedding dimension
HIDDEN_DIM = 128        # LSTM hidden size
NUM_LAYERS = 2          # LSTM layers
DROPOUT = 0.2           # Dropout rate
SEQ_LENGTH = 64         # Training sequence length
BATCH_SIZE = 64         # Batch size
LEARNING_RATE = 0.002   # Initial learning rate
NUM_EPOCHS = 50         # Training epochs
TEMPERATURE = 0.8       # Generation temperature


# =============================================================================
# MODEL DEFINITION
# =============================================================================

_CharLSTMClass = None


def _get_charlstm_class():
    """Factory function to create CharLSTM class after torch import."""
    global _CharLSTMClass
    if _CharLSTMClass is not None:
        return _CharLSTMClass

    lazy_import_torch()

    class CharLSTM(nn.Module):
        """Character-level LSTM for narrative generation."""

        def __init__(self, vocab_size: int, embed_dim: int = EMBED_DIM,
                     hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS,
                     dropout: float = DROPOUT):
            super().__init__()

            self.vocab_size = vocab_size
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            # Embedding layer
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

            # LSTM
            self.lstm = nn.LSTM(
                embed_dim, hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False  # Unidirectional for generation
            )

            # Output projection
            self.fc = nn.Linear(hidden_dim, vocab_size)

            # Dropout
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, hidden=None):
            """Forward pass.

            Args:
                x: Input tensor [batch, seq_len]
                hidden: Optional (h_0, c_0) tuple

            Returns:
                logits: Output logits [batch, seq_len, vocab_size]
                hidden: Updated hidden state
            """
            # Embed
            embedded = self.dropout(self.embedding(x))  # [batch, seq, embed]

            # LSTM
            output, hidden = self.lstm(embedded, hidden)  # [batch, seq, hidden]

            # Project to vocabulary
            logits = self.fc(self.dropout(output))  # [batch, seq, vocab]

            return logits, hidden

        def init_hidden(self, batch_size: int, device):
            """Initialize hidden state."""
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            return (h_0, c_0)

        def generate(self, start_text: str, char_to_idx: Dict[str, int],
                     idx_to_char: Dict[int, str], max_length: int = 100,
                     temperature: float = TEMPERATURE, device=None) -> str:
            """Generate text from a seed string."""
            if device is None:
                device = next(self.parameters()).device

            # Set to inference mode (no gradients, no dropout)
            self.train(False)
            generated = list(start_text)

            # Convert start text to indices
            input_seq = [char_to_idx.get(c, char_to_idx.get("<unk>", 3)) for c in start_text]
            input_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)

            hidden = self.init_hidden(1, device)

            with torch.no_grad():
                # Process seed
                _, hidden = self.forward(input_tensor, hidden)

                # Generate
                current_char = input_seq[-1]
                for _ in range(max_length - len(start_text)):
                    input_tensor = torch.tensor([[current_char]], dtype=torch.long, device=device)
                    logits, hidden = self.forward(input_tensor, hidden)

                    # Apply temperature
                    logits = logits[0, 0] / temperature
                    probs = torch.softmax(logits, dim=0)

                    # Sample
                    current_char = torch.multinomial(probs, 1).item()

                    # Stop at EOS
                    if current_char == char_to_idx.get("<eos>", 2):
                        break

                    char = idx_to_char.get(current_char, "")
                    if char and char not in ("<pad>", "<sos>", "<eos>", "<unk>"):
                        generated.append(char)

            return "".join(generated)

    _CharLSTMClass = CharLSTM
    return _CharLSTMClass


# =============================================================================
# DATA LOADING
# =============================================================================

def load_corpus(corpus_path: Path) -> List[str]:
    """Load training corpus."""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def load_vocabulary(vocab_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load vocabulary mappings."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        char_to_idx = json.load(f)
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    return char_to_idx, idx_to_char


def prepare_sequences(corpus: List[str], char_to_idx: Dict[str, int],
                       seq_length: int = SEQ_LENGTH) -> List[Tuple[List[int], List[int]]]:
    """Prepare training sequences (input, target pairs)."""
    sequences = []

    sos_idx = char_to_idx.get("<sos>", 1)
    eos_idx = char_to_idx.get("<eos>", 2)
    unk_idx = char_to_idx.get("<unk>", 3)

    for text in corpus:
        # Convert to indices with SOS/EOS
        indices = [sos_idx] + [char_to_idx.get(c, unk_idx) for c in text] + [eos_idx]

        # Create overlapping sequences
        for i in range(0, len(indices) - seq_length, seq_length // 2):
            input_seq = indices[i:i + seq_length]
            target_seq = indices[i + 1:i + seq_length + 1]

            if len(input_seq) == seq_length and len(target_seq) == seq_length:
                sequences.append((input_seq, target_seq))

    return sequences


class NarrativeDataset:
    """Dataset for character-level language modeling."""

    def __init__(self, sequences: List[Tuple[List[int], List[int]]]):
        lazy_import_torch()
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long)
        )


def create_dataloader(dataset, batch_size: int = BATCH_SIZE, shuffle: bool = True):
    """Create DataLoader for training."""
    lazy_import_torch()
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device) -> float:
    """Train for one epoch."""
    model.train(True)
    total_loss = 0
    num_batches = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward
        optimizer.zero_grad()
        hidden = model.init_hidden(inputs.size(0), device)
        logits, _ = model(inputs, hidden)

        # Compute loss
        loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def train_model(model, train_loader, val_loader, num_epochs: int = NUM_EPOCHS,
                lr: float = LEARNING_RATE, device=None) -> Dict:
    """Train the CharLSTM model."""
    lazy_import_torch()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 7

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        model.train(False)  # Switch to inference mode
        val_loss = 0
        num_batches = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                hidden = model.init_hidden(inputs.size(0), device)
                logits, _ = model(inputs, hidden)
                loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))
                val_loss += loss.item()
                num_batches += 1
        val_loss /= num_batches

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "models/narrative_model_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return history


# =============================================================================
# EXPORT
# =============================================================================

def export_onnx(model, vocab_size: int, output_path: Path, device=None):
    """Export model to ONNX format using legacy exporter for compatibility."""
    lazy_import_torch()

    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    model.train(False)  # Switch to inference mode

    # Dummy input - static shape for mobile compatibility
    # Batch=1, Seq=64 (fixed for mobile inference)
    dummy_input = torch.randint(0, vocab_size, (1, SEQ_LENGTH), dtype=torch.long, device=device)

    # Export using legacy exporter (dynamo=False) for better compatibility
    # Static shapes are fine for mobile - we process fixed-length chunks
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=14,  # Stable opset with wide ONNX Runtime support
        do_constant_folding=True,
        dynamo=False,  # Use legacy exporter for stability
    )

    # Size check
    size_kb = output_path.stat().st_size / 1024
    print(f"Exported ONNX model: {output_path} ({size_kb:.1f} KB)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Train and export CharLSTM narrative model."""
    lazy_import_torch()

    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    models_dir = script_dir / "models"
    models_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("CharLSTM Narrative Model Training")
    print("=" * 60)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("\n[1/5] Loading corpus and vocabulary...")
    corpus = load_corpus(data_dir / "narrative_corpus.txt")
    char_to_idx, idx_to_char = load_vocabulary(data_dir / "narrative_vocab.json")
    vocab_size = len(char_to_idx)
    print(f"  Corpus: {len(corpus):,} samples")
    print(f"  Vocabulary: {vocab_size} tokens")

    # Prepare sequences
    print("\n[2/5] Preparing training sequences...")
    sequences = prepare_sequences(corpus, char_to_idx)
    random.shuffle(sequences)

    # Split train/val
    split_idx = int(len(sequences) * 0.9)
    train_seqs = sequences[:split_idx]
    val_seqs = sequences[split_idx:]
    print(f"  Train: {len(train_seqs):,} sequences")
    print(f"  Val: {len(val_seqs):,} sequences")

    # Create dataloaders
    train_dataset = NarrativeDataset(train_seqs)
    val_dataset = NarrativeDataset(val_seqs)
    train_loader = create_dataloader(train_dataset, shuffle=True)
    val_loader = create_dataloader(val_dataset, shuffle=False)

    # Create model
    print("\n[3/5] Creating CharLSTM model...")
    CharLSTM = _get_charlstm_class()
    model = CharLSTM(vocab_size)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    print(f"  Estimated size: {num_params * 4 / (1024 * 1024):.2f} MB (FP32)")

    # Train
    print("\n[4/5] Training...")
    history = train_model(model, train_loader, val_loader, device=device)

    # Load best model
    model.load_state_dict(torch.load(models_dir / "narrative_model_best.pt"))

    # Test generation
    print("\n[5/5] Testing generation...")
    test_prompts = [
        "In ages past",
        "The sage spoke",
        "You enter the",
        "The puzzle",
    ]
    for prompt in test_prompts:
        generated = model.generate(prompt, char_to_idx, idx_to_char, max_length=80, device=device)
        print(f"  '{prompt}' -> {generated}")

    # Export
    print("\n" + "=" * 60)
    print("Exporting models...")

    # PyTorch
    torch.save(model.state_dict(), models_dir / "narrative_model.pt")
    pt_size = (models_dir / "narrative_model.pt").stat().st_size / 1024
    print(f"  PyTorch: {pt_size:.1f} KB")

    # ONNX
    onnx_path = models_dir / "narrative_model.onnx"
    export_onnx(model, vocab_size, onnx_path, device=torch.device("cpu"))

    print("\n" + "=" * 60)
    print("Done! Model ready for mobile deployment.")
    print("=" * 60)


if __name__ == "__main__":
    main()
