"""Benign-only autoencoder anomaly scorer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AutoencoderScorer:
    model: Any
    threshold: float
    input_dim: int

    @classmethod
    def fit(cls, X_train_benign, X_validation_benign, *, hidden_dims=(128, 64, 16), epochs=100, patience=10, batch_size=256, learning_rate=1e-3, seed=42, fpr=0.01) -> "AutoencoderScorer":
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to train the autoencoder") from exc
        torch.manual_seed(seed)
        input_dim = int(X_train_benign.shape[1])
        model = _TorchAutoencoder(input_dim, hidden_dims)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        train_loader = DataLoader(TensorDataset(torch.as_tensor(X_train_benign, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
        validation = torch.as_tensor(X_validation_benign, dtype=torch.float32)
        best_state = None
        best_loss = float("inf")
        stale = 0
        for _ in range(epochs):
            model.train()
            for (batch,) in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(model(batch), batch)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = float(loss_fn(model(validation), validation))
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        scores = _torch_scores(model, X_validation_benign)
        threshold = float(np.quantile(scores, 1.0 - fpr)) if len(scores) else 0.0
        return cls(model=model, threshold=threshold, input_dim=input_dim)

    def score(self, X) -> np.ndarray:
        return _torch_scores(self.model, X)

    def predict(self, X) -> np.ndarray:
        return (self.score(X) >= self.threshold).astype("int8")

    def save(self, path: str):
        import torch

        torch.save({"state_dict": self.model.state_dict(), "threshold": self.threshold, "input_dim": self.input_dim}, path)


class _TorchAutoencoder:
    def __init__(self, input_dim: int, hidden_dims):
        from torch import nn

        encoder_layers = []
        previous = input_dim
        for width in hidden_dims:
            encoder_layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        decoder_layers = []
        for width in reversed(hidden_dims[:-1]):
            decoder_layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        decoder_layers.append(nn.Linear(previous, input_dim))
        self._module = nn.Sequential(*(encoder_layers + decoder_layers))

    def parameters(self):
        return self._module.parameters()

    def train(self, mode=True):
        self._module.train(mode)
        return self

    def eval(self):
        self._module.eval()
        return self

    def __call__(self, values):
        return self._module(values)

    def state_dict(self):
        return self._module.state_dict()

    def load_state_dict(self, state):
        return self._module.load_state_dict(state)


def _torch_scores(model, X) -> np.ndarray:
    import torch

    values = torch.as_tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        reconstruction = model(values)
        return torch.mean((reconstruction - values) ** 2, dim=1).cpu().numpy()
