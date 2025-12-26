import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import optuna
import os


class VariableSizeDataset(Dataset):
    def __init__(self, data_path, max_size=20):
        self.max_size = max_size
        self.data = np.load(data_path)
        self.all_keys = [k for k in self.data.files if k.startswith("size")]
        self.samples = []
        for key in self.all_keys:
            grids = self.data[key]
            size = int(key.replace("size", ""))
            for g in grids:
                self.samples.append((g, size))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        grid, size = self.samples[idx]
        # Flatten and pad to max_size^2 for batching simplicity,
        # but model will only look at sequence length size^2.
        target = np.zeros(self.max_size * self.max_size, dtype=np.int64)
        target[: size * size] = grid.flatten()

        # Masking logic
        inp = target.copy()
        mask = np.random.rand(size * size) < 0.7
        inp[: size * size][mask] = 0

        return torch.from_numpy(inp), torch.from_numpy(target), torch.tensor(size)


class KeenTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, nlayers=6, max_size=20):
        super().__init__()
        self.max_size = max_size
        self.embedding = nn.Embedding(21, d_model)  # 0=mask, 1-20=digits
        self.pos_encoding = nn.Parameter(torch.randn(1, max_size * max_size, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.fc_out = nn.Linear(d_model, 21)

    def forward(self, x, size):
        B, S = x.shape
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :S, :]

        # Create mask for padding
        # In this dataset, we treat sequence as size*size.
        # Batching might require padding masks if sizes are mixed.
        # For simplicity in this iteration, we use batch size 1 or same-size batches.

        x = self.transformer(x)
        return self.fc_out(x)


def train_keen(trial):
    # Param sweep items
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    nhead = trial.suggest_categorical("nhead", [4, 8])
    nlayers = trial.suggest_int("nlayers", 3, 8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeenTransformer(d_model=d_model, nhead=nhead, nlayers=nlayers).to(device)

    # Load dataset
    dataset = VariableSizeDataset("data/latin_squares_massive.npz")
    # Use a small subset for sweep
    subset_indices = torch.randperm(len(dataset))[:5000]
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, subset_indices), batch_size=32, shuffle=True
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(2):  # Short sweep
        for inp, target, size in train_loader:
            inp, target = inp.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(inp, size)  # [B, S, C]
            loss = criterion(output.transpose(1, 2), target)
            loss.backward()
            optimizer.step()

    return loss.item()


if __name__ == "__main__":
    if os.path.exists("data/latin_squares_massive.npz"):
        study = optuna.create_study(direction="minimize")
        study.optimize(train_keen, n_trials=10)
        print("Best Params:", study.best_params)
    else:
        print("Dataset missing. Run generation first.")
