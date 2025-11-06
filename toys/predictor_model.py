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

    THE BIG IDEA:
    =============

    Input: 24D vector (chunk_i)
    Output: 24D vector (predicted chunk_i+1)

    The network is a simple transformation: take semantic input, transform it through
    layers of thinking, output the predicted next semantic state.

    WHY MULTIPLE LAYERS?
    ====================

    Single layer: LINEAR transformation only
    - Fast but limited: can only rotate/scale the space
    - Like: can only change the direction, not learn complex patterns

    Multiple layers: LINEAR → NONLINEAR → LINEAR → NONLINEAR → LINEAR
    - Slower but more powerful: can learn curved relationships
    - Like: can learn "if concept A increases, concept B decreases in a complex way"

    The ReLU activation function:
    - Linear is flat: f(x) = ax + b
    - ReLU bends it: f(x) = max(0, x)  (dead below zero, linear above)
    - This bend lets the network learn nonlinear patterns

    ARCHITECTURE:
    ==============

    Layer 1:   24D → 48D   (expand: "think about more aspects")
               ReLU        (nonlinear: "think creatively")

    Layer 2:   48D → 24D   (compress: "synthesize back to core concepts")
               ReLU        (nonlinear again)

    Layer 3:   24D → 24D   (final output)
               (no ReLU: let output be any value)

    TOTAL PARAMETERS: ~2,400 (very small, intentionally)

    WHY SO SMALL?
    ==============
    - We're training on small texts (100-50,000 chunks)
    - Bigger network = overfits (memorizes instead of learning pattern)
    - Smaller network = learns the essential semantic flow

    ANALOGY:
    ========
    Imagine teaching someone to write poetry:
    - A huge brain might memorize all examples (bad)
    - A small brain must extract the essence (good)

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

        # Create training data and train
        input_chunks, target_chunks = self._prepare_chunk_pairs(latent_chunks)
        history = self._train_with_early_stopping(input_chunks, target_chunks, epochs, batch_size, patience)
        return history

    def _prepare_chunk_pairs(self, latent_chunks: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create (chunk_i, chunk_i+1) training pairs.

        WHAT'S HAPPENING:
        =================

        You have a sequence of text chunks, each represented as a 24D vector.

        chunk_0 → chunk_1 → chunk_2 → chunk_3 → ... → chunk_N

        We create training examples:
        - Input: chunk_0,  Target: chunk_1 (predict the next chunk)
        - Input: chunk_1,  Target: chunk_2
        - Input: chunk_2,  Target: chunk_3
        - ... and so on

        WHY?
        ====
        The network learns: "given chunk_i's semantic meaning, what would chunk_i+1 likely be?"

        This teaches the network the DIRECTION of semantic flow in this person's writing.

        Think of it like:
        - Completing sentences: "The sun is ___" → "bright"
        - Finding patterns: In Austen's writing, discussions of society lead to...?
        - Learning the rhythm: Melville's obsession, Woolf's fragmentation

        THE RESULT:
        ===========
        After training, the network has learned this persona's semantic "next step."
        When you ask it a question, it predicts what that persona would say next.
        """
        chunk_data = torch.FloatTensor(latent_chunks).to(self.device)
        input_chunks = chunk_data[:-1]   # all but last: chunk_0, chunk_1, ... chunk_N-1
        target_chunks = chunk_data[1:]   # all but first: chunk_1, chunk_2, ... chunk_N
        return input_chunks, target_chunks

    def _train_with_early_stopping(
        self,
        input_chunks: torch.Tensor,
        target_chunks: torch.Tensor,
        epochs: int,
        batch_size: int,
        patience: int,
    ) -> dict:
        """Train the predictor with early stopping.

        WHAT'S HAPPENING:
        =================

        We're teaching the network to predict: given chunk_i (input), output chunk_i+1 (target).

        TRAINING LOOP (simplified):
        ===========================

        For each epoch (1 to max_epochs):
          1. Shuffle data into mini-batches
          2. For each batch:
             a. Feed batch through network: prediction = network(input)
             b. Calculate error: loss = distance(prediction, target)
             c. Improve network weights to reduce error
             d. Repeat

          3. After epoch, check: did we improve?
             - If yes: reset patience counter
             - If no: increment patience counter
             - If patience runs out: stop (we've learned what we can)

        MATH-FREE VERSION:
        ==================
        Think of it like teaching someone to write in a style:

        1. Show them examples: (Austen sentence) → (Austen's next sentence)
        2. They try: guess what comes next
        3. You tell them if they're right or wrong
        4. They learn from mistakes
        5. Eventually they can continue the style
        6. When they stop improving, they've learned the style

        THE HYPERPARAMETERS:
        ====================
        - epochs: max training iterations
        - batch_size: how many examples per training step (4 = learn from 4 examples at once)
        - patience: how many epochs without improvement before giving up

        WHY SMALL BATCHES?
        ==================
        Batch size 4 means we look at 4 examples before updating weights.
        This is more stable than looking at 1 example at a time.
        But not so many that we lose flexibility.
        """
        best_loss = float("inf")
        patience_counter = 0
        history = {"loss": []}

        print(f"📚 Training predictor on {len(input_chunks)} chunk pairs...")
        print(f"   Device: {self.device.upper()}")
        print(f"   Batch size: {batch_size}, Max epochs: {epochs}, Patience: {patience}")
        print()

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            # Process mini-batches
            for i in range(0, len(input_chunks), batch_size):
                batch_input = input_chunks[i : i + batch_size]
                batch_target = target_chunks[i : i + batch_size]

                # Forward pass: compute prediction
                pred = self.model(batch_input)

                # Compute error
                loss = self.criterion(pred, batch_target)

                # Backward pass: update weights to reduce error
                self.optimizer.zero_grad()  # Clear old gradients
                loss.backward()              # Compute new gradients
                self.optimizer.step()        # Update weights

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            history["loss"].append(avg_loss)

            # Progress update every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1:3d}/{epochs}: loss = {avg_loss:.6f}")

            # Early stopping: if we're not improving, stop training
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   → Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
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
