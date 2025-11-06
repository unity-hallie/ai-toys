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
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        chunks = []
        current_chunk = []
        current_length = 0

        for sent in sentences:
            words = sent.split()
            if current_length + len(words) > chunk_size and current_chunk:
                chunks.append(". ".join(current_chunk) + ".")
                current_chunk = [sent]
                current_length = len(words)
            else:
                current_chunk.append(sent)
                current_length += len(words)

        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")

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

        # Read and split
        print(f"📖 Reading {text_path}...")
        with open(text_path) as f:
            text = f.read()

        chunks = self._split_into_chunks(text)
        print(f"📝 Split into {len(chunks)} chunks")

        # Embed
        print(f"🧠 Embedding {len(chunks)} chunks to 384D...")
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True)
        print(f"   Shape: {embeddings.shape}")

        # Fit PCA
        print(f"🔄 Fitting PCA to {self.output_dim}D (Leech lattice)...")
        self.pca = PCA(n_components=self.output_dim)
        self.pca.fit(embeddings)

        explained_variance = self.pca.explained_variance_ratio_.sum()
        print(f"   Explained variance: {explained_variance:.4f}")
        print(f"   Variance per component: {self.pca.explained_variance_ratio_}")

        # Save
        with open(output_path, "wb") as f:
            pickle.dump(self.pca, f)
        print(f"💾 Saved to {output_path}")

        return output_path

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
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        trainer = PCATrainer()
        trainer.pca = pca
        return trainer
