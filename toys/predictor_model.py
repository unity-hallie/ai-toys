"""
Predictor Model - Small 24D → 24D neural network for chunk prediction.

ARCHITECTURE:
=============
Input:  24D (chunk_i projected through PCA)
        ↓
Hidden: Linear(24 → 48) + ReLU
        ↓
Hidden: Linear(48 → 24) + ReLU
        ↓
Output: Linear(24 → 24) (chunk_i+1 predicted)

WHY THIS SIZE?
==============
- Cheap: Only 24×48 + 48×24 + 24×24 = 2400 parameters
- Fast: Single forward pass ~microseconds
- Learnable: Can fit on small text (20-40 chunks)
- Task-specific: Optimized for predicting next chunk in 24D space
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple
import numpy as np


class PredictorNet(nn.Module):
    """Small 24D → 24D predictor for chunk sequences."""

    def __init__(self, latent_dim: int = 24):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, 24) tensor of latent vectors

        Returns:
            (batch, 24) tensor of predicted next latents
        """
        return self.network(x)


class PredictorTrainer:
    """Train and manage predictor models."""

    def __init__(
        self,
        latent_dim: int = 24,
        learning_rate: float = 0.001,
        device: str = None,
    ):
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate

        # Auto-detect best device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        self.model = PredictorNet(latent_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def fit(
        self,
        latent_chunks: np.ndarray,
        epochs: int = 50,
        batch_size: int = 4,
        patience: int = 10,
    ) -> dict:
        """
        Train predictor on latent chunks.

        Args:
            latent_chunks: (n_chunks, 24) array of PCA-projected chunks
            epochs: Number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience

        Returns:
            Dictionary with training info
        """
        n_chunks = len(latent_chunks)
        if n_chunks < 2:
            raise ValueError("Need at least 2 chunks to train")

        # Create (chunk_i, chunk_i+1) pairs
        chunk_data = torch.FloatTensor(latent_chunks).to(self.device)
        input_chunks = chunk_data[:-1]  # chunk_i
        target_chunks = chunk_data[1:]  # chunk_i+1

        best_loss = float("inf")
        patience_counter = 0
        history = {"loss": []}

        print(f"📚 Training predictor on {len(input_chunks)} chunk pairs...")
        for epoch in range(epochs):
            # Mini-batch training
            total_loss = 0
            for i in range(0, len(input_chunks), batch_size):
                batch_input = input_chunks[i : i + batch_size]
                batch_target = target_chunks[i : i + batch_size]

                self.optimizer.zero_grad()
                pred = self.model(batch_input)
                loss = self.criterion(pred, batch_target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (len(input_chunks) // batch_size + 1)
            history["loss"].append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs}: loss = {avg_loss:.6f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   Early stopping at epoch {epoch+1}")
                    break

        print(f"✅ Training complete. Final loss: {best_loss:.6f}")
        return history

    def predict(self, latent_vector: np.ndarray) -> np.ndarray:
        """
        Predict next chunk from latent vector.

        Args:
            latent_vector: (24,) latent embedding

        Returns:
            (24,) predicted next latent
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(latent_vector).unsqueeze(0).to(self.device)
            pred = self.model(x)
            return pred[0].cpu().numpy()

    def save(self, path: str) -> Path:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"💾 Model saved to {path}")
        return path

    @staticmethod
    def load(path: str, device: str = "cpu") -> "PredictorTrainer":
        """Load model from disk."""
        trainer = PredictorTrainer(device=device)
        trainer.model.load_state_dict(torch.load(path, map_location=device))
        trainer.model.eval()
        return trainer
