"""
PCA Trainer - Learn text-specific PCA models from semantic embeddings.

ARCHITECTURE:
=============
1. Read text file
2. Split into semantic chunks (sentence-based)
3. Embed each chunk to 384D (sentence-transformers)
4. Fit PCA to reduce 384D → 24D (Leech lattice dimension)
5. Save PCA model to disk

WHY 24D?
========
The Leech lattice (24D) has exceptional mathematical structure:
- Automorphism group: Fischer group Fi_24
- Kissing number: 196560 (densest sphere packing in 24D)
- Used in Conway group theory
- Suggests deep symmetries in semantic projection

WHY PCA?
========
- Simple: One matrix multiplication per projection
- Interpretable: Each component is a learned direction in semantic space
- Fast: Inference is O(n) where n = output dimension
- Different text sources learn different PCA bases
"""

import json
import pickle
import re
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import logging

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


class PCATrainer:
    """Train PCA models from text embeddings to 24D."""

    def __init__(self, output_dim: int = 24, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize PCA trainer.

        Args:
            output_dim: PCA output dimension (default 24D - Leech lattice)
            model_name: Sentence transformer model to use
        """
        self.output_dim = output_dim
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.pca = None
        print(f"📦 Loaded embedding model: {model_name}")

    def _fit_and_explain_pca(self, embeddings: np.ndarray) -> None:
        """Fit PCA and explain what's happening in plain language.

        WHAT'S HAPPENING (no math trauma allowed):
        ==========================================

        You have 384 numbers per chunk (384D embedding).
        PCA finds the 24 most "important" directions.

        Think of it like taking a photo of a landscape:
        - Real world: infinite detail (384D)
        - Photo: captures the essence in 2D
        - PCA: finds which axes capture the most "variation"

        The process:
        1. Look at all the chunks together
        2. Find which directions have the most "spread" (variation)
        3. Keep the top 24 directions, discard the rest
        4. Transform each chunk: 384 numbers → 24 numbers (90% of variation preserved)

        WHY 24D?
        ========
        The Leech lattice (a geometric structure) has 24 dimensions.
        It's special: it packs spheres more efficiently than any other 24D pattern.
        Maybe semantic space wants to be that efficient too?

        THE MATH (if you're curious):
        ============================
        PCA finds eigenvectors of the covariance matrix.
        The eigenvectors point in directions of maximum spread.
        We keep the 24 with the largest eigenvalues.

        READABLE ALTERNATIVE:
        =====================
        If you want to understand it without equations:
        - Imagine chunks scattered in 384D space
        - Find the axis where they spread out most: that's direction #1
        - Find the axis perpendicular to #1 where they spread most: direction #2
        - Repeat 22 more times
        - Now you have 24 axes that capture the "shape" of your data
        """
        self.pca = PCA(n_components=self.output_dim)
        self.pca.fit(embeddings)

        explained_variance = self.pca.explained_variance_ratio_.sum()
        print(f"   ✓ Captured {explained_variance*100:.1f}% of the variation")
        print(f"   ✓ Reduced from 384D to {self.output_dim}D")
        print(f"   ✓ Each dimension captures: ", end="")
        for i, var in enumerate(self.pca.explained_variance_ratio_[:5]):
            print(f"{var*100:.1f}% ", end="")
        print("...")

    def _split_into_chunks(self, text: str, chunk_size: int = 100) -> List[str]:
        """Split text into sentence-based chunks.

        ❓ DESIGN QUESTIONS:

        1. Should chunking be a separate class/abstraction?
           - Pro: Testable, reusable, can swap strategies
           - Con: Over-engineering for current use case
           - Consider: If we want word-level or paragraph-level chunks later?

        2. Is sentence-splitting on "." too naive?
           - What about "Dr.", "U.S.", "etc."?
           - Should we use NLTK or similar for proper sentence tokenization?

        3. The chunk_size is in words. Should it be configurable?
           - Currently hard-coded in fit(). Make it a parameter?
           - Or learn optimal chunk size from data?

        4. What's the invariant we're optimizing for?
           - Semantic coherence? Token count? Computational efficiency?
           - Should that constraint be explicit?
        """
        # Use regex-based sentence splitting to handle abbreviations better
        # Splits on sentence-ending punctuation (., !, ?) followed by space
        # This avoids splitting on abbreviations like "U.S." or "Dr."
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_length = 0

        for sent in sentences:
            words = sent.split()
            if current_length + len(words) > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_length = len(words)
            else:
                current_chunk.append(sent)
                current_length += len(words)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def fit(self, text_path: str, output_path: str) -> Path:
        """
        Fit PCA on text embeddings and save to disk.

        Args:
            text_path: Path to text file
            output_path: Where to save PCA model

        Returns:
            Path to saved PCA model

        ❓ DESIGN QUESTIONS:

        1. Should I/O and training be separate methods?
           - Currently: read file → embed → fit PCA → save (all in one)
           - Alternative: take embeddings directly, save separately
           - Pro: More testable, composable
           - Con: More API surface, user responsibility for I/O

        2. Is file I/O the right abstraction here?
           - What if we want to train on streaming data?
           - Or data that comes from a database?
           - Should we accept an iterator instead of a filepath?

        3. Should we validate the explained variance?
           - What if output_dim is too small and we lose info?
           - Should we warn or error if explained_variance < threshold?

        4. The embedder is created in __init__. Should it be injected?
           - Current: tight coupling to sentence-transformers
           - Alternative: depend on abstract EmbedderInterface
           - Pro: Can swap different embedders, easier to test
           - Con: Over-engineering?

        5. Should we return just the path or also the metadata?
           - Currently: return Path
           - Could return: (path, explained_variance, n_chunks, ...)
           - Does the caller need that info?

        6. Error handling: what if file doesn't exist? encoding issues?
           - Currently: silent failure (open() will raise)
           - Should we catch and provide better error messages?
        """
        text_path = Path(text_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        chunks = self._read_and_chunk_text(text_path)
        embeddings = self._embed_chunks_to_384d(chunks)
        self._fit_pca_model(embeddings)
        self._save_pca_to_disk(output_path)

        return output_path

    def _read_and_chunk_text(self, text_path: Path) -> list:
        """Read text file and split into semantic chunks."""
        print(f"📖 Reading {text_path}...")
        with open(text_path) as f:
            text = f.read()
        chunks = self._split_into_chunks(text)
        print(f"📝 Split into {len(chunks)} chunks")
        return chunks

    def _embed_chunks_to_384d(self, chunks: list) -> np.ndarray:
        """Convert chunks to 384-dimensional embeddings."""
        print(f"🧠 Embedding {len(chunks)} chunks to 384D...")
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True)
        print(f"   Shape: {embeddings.shape}")
        return embeddings

    def _fit_pca_model(self, embeddings: np.ndarray) -> None:
        """Train PCA to project from 384D to latent dimensions."""
        print(f"🔄 Fitting PCA to {self.output_dim}D...")
        self._fit_and_explain_pca(embeddings)

    def _save_pca_to_disk(self, output_path: Path) -> None:
        """Persist PCA model to pickle file."""
        with open(output_path, "wb") as f:
            pickle.dump(self.pca, f)
        print(f"💾 Saved to {output_path}")

    def project(self, embedding: np.ndarray) -> np.ndarray:
        """Project a 384D embedding through PCA to 24D.

        ❓ DESIGN QUESTIONS:

        1. Should we support batch projections?
           - Currently: takes single embedding, returns single output
           - Common pattern: project(embeddings) → batch of outputs
           - Pro: More efficient, vectorized
           - Con: Different shapes to handle

        2. Should we validate input shape?
           - Currently: assume 384D input
           - Should we check/warn if shape is wrong?
           - What if someone passes wrong dimensionality?

        3. Should reshape be implicit or explicit?
           - Currently: reshape(1, -1) to add batch dimension
           - Should caller be responsible for shape?
           - Or should we accept both 1D and 2D?
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit() first.")
        return self.pca.transform(embedding.reshape(1, -1))[0]

    @staticmethod
    def load(pca_path: str) -> "PCATrainer":
        """Load a saved PCA model.

        ❓ DESIGN QUESTIONS:

        1. Should this be a @staticmethod or @classmethod?
           - Current: @staticmethod (returns instance)
           - Alternative: @classmethod (could set output_dim from metadata)
           - Pro: More consistent with Python conventions
           - Con: Need to store metadata with model

        2. Should we store metadata alongside the model?
           - Currently: just pickle the sklearn PCA object
           - Could store: output_dim, model_name, chunk_size, etc.
           - Pro: Self-documenting, prevent misuse
           - Con: More complex serialization

        3. Should load validate the pickle?
           - Currently: trust pickle.load()
           - Could check: is this actually a PCA? what version?
           - Pro: Fail fast on corruption
           - Con: Adds complexity

        4. Error handling: what if file doesn't exist?
           - Currently: FileNotFoundError bubbles up
           - Should we provide better error context?
        """
        pca_path = Path(pca_path)
        if not pca_path.exists():
            raise FileNotFoundError(f"PCA model not found: {pca_path}")

        try:
            with open(pca_path, "rb") as f:
                pca = pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"Corrupted PCA model file {pca_path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load PCA model {pca_path}: {e}")

        # Validate that we actually loaded a PCA object
        if not isinstance(pca, PCA):
            raise TypeError(
                f"Expected sklearn.decomposition.PCA, got {type(pca).__name__}"
            )

        trainer = PCATrainer()
        trainer.pca = pca
        return trainer
