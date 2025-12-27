import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os
import math


class MassiveLatinSquareDataset(Dataset):
    """
    Dataset generator for grids from 3x3 to 16x16.
    """

    def __init__(self, data_path=None, max_size=16):
        self.max_size = max_size
        self.samples = []
        if data_path and os.path.exists(data_path):
            data = np.load(data_path)
            for key in data.files:
                if not key.startswith("size"):
                    continue
                size = int(key.replace("size", ""))
                if size < 3 or size > max_size:
                    continue
                grids = data[key]
                for g in grids:
                    self.samples.append((g, size))

    def __len__(self):
        return len(self.samples) if self.samples else 10000

    def __getitem__(self, idx):
        if not self.samples:
            size = np.random.randint(3, self.max_size + 1)
            grid = np.zeros((self.max_size, self.max_size))
            return torch.from_numpy(grid).long(), torch.tensor(size).long()

        grid, size = self.samples[idx % len(self.samples)]
        padded = np.zeros((self.max_size, self.max_size), dtype=np.int64)
        padded[:size, :size] = grid

        target = torch.from_numpy(padded).long()
        mask = torch.rand(padded.shape) < 0.7
        inp = target.clone()
        inp[mask] = 0
        inp[size:, :] = 0
        inp[:, size:] = 0

        return inp, target, torch.tensor(size).long()


class RelationalTransformer(nn.Module):
    """
    Transformer-based solver for variable sized grids up to 16x16.
    """

    def __init__(self, max_size=16, num_classes=17, d_model=128, nhead=8, num_layers=6):
        super().__init__()
        self.max_size = max_size
        self.embedding = nn.Embedding(num_classes, d_model)
        # Proper initialization for positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_size * max_size, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, H, W]
        B, H, W = x.shape
        x = x.view(B, -1)  # Flatten to sequence: [B, H*W]

        x = self.embedding(x)  # [B, Seq, D]
        x = x + self.pos_embed[:, : H * W, :]

        x = self.transformer(x)
        x = self.output_head(x)

        return x.view(B, H, W, -1).permute(0, 3, 1, 2)  # [B, Class, H, W]


def get_gpu_info():
    """Detect GPU capabilities for optimal compile settings."""
    if not torch.cuda.is_available():
        return {"name": "CPU", "sm_count": 0, "compute_cap": (0, 0)}
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "sm_count": props.multi_processor_count,
        "compute_cap": (props.major, props.minor),
    }


def train_massive():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info['name']} ({gpu_info['sm_count']} SMs)")
    print(f"Training Model (3x3 to 16x16) on {device} (AMP: {use_amp})...")

    # Enable TF32 for tensor cores (significant speedup on Ampere+)
    if device.type == "cuda" and gpu_info["compute_cap"][0] >= 8:
        torch.set_float32_matmul_precision("high")
        print("TF32 enabled for tensor cores.")

    # Load and split dataset (80/20 train/val)
    full_dataset = MassiveLatinSquareDataset("data/latin_squares_massive.npz", max_size=16)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = RelationalTransformer(max_size=16, num_classes=17).to(device)

    # Use torch.compile with appropriate mode based on GPU SMs
    # max-autotune requires 80+ SMs; use reduce-overhead for smaller GPUs
    if hasattr(torch, "compile") and device.type == "cuda":
        compile_mode = "max-autotune" if gpu_info["sm_count"] >= 80 else "reduce-overhead"
        print(f"torch.compile mode: {compile_mode}")
        model = torch.compile(model, mode=compile_mode)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_loss = float("inf")
    epochs = 10

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for i, (inp, target, size) in enumerate(train_loader):
            inp, target = inp.to(device), target.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(inp)
                loss = criterion(output, target)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch} [{i}] Loss: {loss.item():.4f}")

        # Validation phase
        model.train(False)
        val_loss = 0.0
        with torch.no_grad():
            for inp, target, size in val_loader:
                inp, target = inp.to(device), target.to(device)
                output = model(inp)
                val_loss += criterion(output, target).item()

        val_loss /= max(len(val_loader), 1)
        print(f"Epoch {epoch} Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "keen_solver_16x16_best.pth")

        scheduler.step()

    # Save model
    torch.save(model.state_dict(), "keen_solver_16x16.pth")
    print("Model saved.")

    # Export to ONNX
    print("Exporting to ONNX...")
    model.eval()
    dummy_input = torch.zeros(1, 16, 16).long().to(device)
    torch.onnx.export(
        model,
        dummy_input,
        "keen_solver_16x16.onnx",
        input_names=["input_grid"],
        output_names=["cell_logits"],
    )
    print("ONNX Export complete: keen_solver_16x16.onnx")


if __name__ == "__main__":
    train_massive()
