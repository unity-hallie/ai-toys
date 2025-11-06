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
    """Small 24D → 24D predictor for chunk sequences.

    ❓ ARCHITECTURE QUESTIONS:

    1. Is this network size right?
       - Current: 24 → 48 → 24 → 24 (2400 params)
       - Trade-off: Too small = underfitting, too large = overfitting
       - How would you tune this for different text lengths?

    2. Should the hidden layer be latent_dim * 2?
       - Why not 128? 256? half?
       - Is this a principled choice or arbitrary?

    3. Should the last layer have activation?
       - Current: No activation after final 24 → 24
       - Alternative: Add ReLU to constrain output?
       - Does that make semantic sense?

    4. Is residual connection useful here?
       - Current: pure sequential, no shortcuts
       - Could try: x_out = x + network(x)
       - Pro: Helps with gradient flow, identity mapping
       - Con: Assumes input ≈ output, true here?

    5. Should we use dropout for regularization?
       - Current: No dropout
       - Alternative: Add dropout after each ReLU
       - Does it help prevent overfitting on small texts?
    """

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
        """
        ❓ INITIALIZATION QUESTIONS:

        1. Should optimizer and loss be injectable?
           - Current: hardcoded Adam + MSELoss
           - Alternative: pass optimizer_class, loss_fn as args
           - Pro: Flexible, testable, experiment-friendly
           - Con: More API surface
           - When would you want different optimizer?

        2. Should we store hyperparameters as instance variables?
           - Currently: only latent_dim, learning_rate, device stored
           - Could store: batch_size, patience, epochs as defaults
           - Pro: More reproducible, easier to debug
           - Con: More state to manage

        3. Should device auto-detection be a separate function?
           - Current: inline if/elif chain
           - Alternative: extract to get_best_device()
           - Pro: Testable, reusable, clearer intent
           - Con: Premature abstraction?

        4. Should we validate hyperparameters?
           - Currently: no checks
           - Could validate: latent_dim > 0, learning_rate > 0, etc.
           - Pro: Fail fast
           - Con: Runtime checks, adds noise
        """
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

        ❓ TRAINING QUESTIONS:

        1. Should we use validation set?
           - Currently: no train/val split, evaluate on training loss
           - Alternative: hold out 20% for validation
           - Pro: More realistic generalization estimate
           - Con: Small datasets get smaller
           - What's the minimum data size before this matters?

        2. The loss calculation looks suspicious. (epoch + 1) % 10 == 0?
           - Should we log more frequently? less?
           - Should frequency be configurable?
           - Should we log metrics to file?

        3. Is early stopping the right regularization strategy?
           - Current: stop when loss stops improving
           - Alternative: L1/L2 regularization
           - Alternative: Dropout in the network
           - What's appropriate for small texts?

        4. Should we shuffle the data?
           - Currently: sequential pairs (chunk_0→1, 1→2, 2→3, ...)
           - Alternative: sample random pairs
           - Does order matter for semantic chunks?

        5. Are the loss and optimization working?
           - Current: MSELoss (minimize distance)
           - Alternative: Could we do contrastive learning?
           - Alternative: Cosine similarity loss?
           - Which metric actually matters for the downstream task?

        6. Should we normalize the latent vectors before training?
           - Currently: raw PCA outputs
           - Alternative: normalize to unit norm
           - Pro: More stable gradients
           - Con: Loses magnitude information
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
