"""
Magic 8 Ball - Decision oracle using PCA + Predictor models.

ARCHITECTURE:
=============
1. Load PCA model for persona (384D → 24D projection)
2. Load Predictor model for persona (24D → 24D response)
3. Embed question to 384D
4. Project through PCA to 24D
5. Pass through predictor to get predicted response in 24D
6. Match to 10 generic responses (also encoded through same PCA)
7. Return closest match

WHY THIS APPROACH?
==================
- Each persona has learned different PCA basis from their text
- Predictor learns what that persona's semantic "next step" is
- Direct, interpretable, cheap computation
- 24D Leech lattice space for potential symmetry properties
"""

import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple
from sentence_transformers import SentenceTransformer
from toys.predictor_model import PredictorTrainer
import logging

# Global embedder cache - avoid reloading the same model for each persona
_embedder_cache: Dict[str, SentenceTransformer] = {}


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Get or create cached embedder instance."""
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = SentenceTransformer(model_name)
    return _embedder_cache[model_name]

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


class Magic8Ball:
    """Decision oracle using PCA + Predictor models."""

    # 10 generic decision responses (no pronouns)
    RESPONSES = [
        "Yes, absolutely",
        "Yes, but proceed cautiously",
        "Probably yes",
        "Uncertain",
        "Probably not",
        "No, not yet",
        "No, fundamentally wrong",
        "Revisit later",
        "Needs more discussion",
        "Too risky",
    ]

    def __init__(self, persona_name: str, models_dir: str = "toys_models"):
        """
        Initialize magic 8 ball with persona models.

        Args:
            persona_name: Name of persona (must have trained models)
            models_dir: Directory containing trained PCA + Predictor models

        ❓ INITIALIZATION QUESTIONS:

        1. Should the embedder be configurable?
           - Currently: hardcoded "all-MiniLM-L6-v2"
           - Alternative: pass embedder or model_name as argument
           - Pro: More flexible, support different embeddings
           - Con: More API surface
           - Does this need to match the PCA training embedding?

        2. Should we cache the embedder like we cache responses?
           - Currently: create new SentenceTransformer every time
           - Alternative: Use a singleton or class variable
           - Pro: Faster initialization on second Magic8Ball
           - Con: Shared state, harder to test

        3. Should we validate that PCA and predictor match?
           - Currently: load both and hope they're compatible
           - Alternative: check metadata (latent_dim, etc.)
           - Pro: Fail fast on mismatch
           - Con: Requires storing metadata with models

        4. Should response embedding happen lazily or eagerly?
           - Currently: eager (during __init__)
           - Alternative: lazy (during first consult)
           - Pro: Faster initialization
           - Con: First query is slower

        5. Should we support custom responses?
           - Currently: RESPONSES are hardcoded class variable
           - Alternative: pass responses_list to __init__
           - Pro: More flexible, persona-specific responses
           - Con: More complex initialization
        """
        self.persona_name = persona_name
        self.models_dir = Path(models_dir)
        self.embedder = get_embedder("all-MiniLM-L6-v2")

        self.pca = self._load_pca_model(persona_name)
        self.predictor = self._load_predictor_model(persona_name)
        self._validate_model_compatibility()
        self.response_vectors_24d = self._precompute_response_embeddings()

    def _load_pca_model(self, persona_name: str):
        """Load and validate PCA model from disk."""
        pca_path = self.models_dir / f"pca_{persona_name}.pkl"
        if not pca_path.exists():
            raise FileNotFoundError(f"PCA model not found: {pca_path}")
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        print(f"✅ Loaded PCA model: {pca_path}")
        return pca

    def _load_predictor_model(self, persona_name: str):
        """Load predictor model with automatic device detection."""
        predictor_path = self.models_dir / f"predictor_{persona_name}.pt"
        if not predictor_path.exists():
            raise FileNotFoundError(f"Predictor not found: {predictor_path}")
        device = self._detect_device()
        predictor = PredictorTrainer.load(str(predictor_path), device=device)
        print(f"✅ Loaded Predictor model: {predictor_path} (device: {device})")
        return predictor

    def _detect_device(self) -> str:
        """Detect available compute device: CUDA > MPS > CPU."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _validate_model_compatibility(self) -> None:
        """Ensure PCA output dimension matches predictor input dimension."""
        pca_output_dim = self.pca.n_components_
        predictor_input_dim = self.predictor.latent_dim
        if pca_output_dim != predictor_input_dim:
            raise ValueError(
                f"Dimension mismatch: PCA outputs {pca_output_dim}D but "
                f"Predictor expects {predictor_input_dim}D. Models may be incompatible."
            )

    def _precompute_response_embeddings(self) -> np.ndarray:
        """Precompute all response vectors in latent space."""
        print(f"📚 Embedding {len(self.RESPONSES)} response options...")
        response_embeddings_384d = self.embedder.encode(
            self.RESPONSES, convert_to_numpy=True
        )
        response_vectors_24d = self.pca.transform(response_embeddings_384d)
        print(f"   Shape: {response_vectors_24d.shape}")
        return response_vectors_24d

    def _compute_cosine_similarity(
        self, vector_a: np.ndarray, vector_b: np.ndarray
    ) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        # Add small epsilon to prevent division by zero
        return dot_product / (norm_a * norm_b + 1e-8)

    def _rank_responses_by_similarity(
        self, predicted_vector: np.ndarray
    ) -> list:
        """Score and rank all responses by similarity to predicted vector."""
        similarities = {}
        for i, response_text in enumerate(self.RESPONSES):
            response_vector = self.response_vectors_24d[i]
            similarity = self._compute_cosine_similarity(predicted_vector, response_vector)
            similarities[response_text] = similarity

        # Sort by similarity (descending)
        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    def _find_best_response(
        self, predicted_24d: np.ndarray
    ) -> Tuple[str, float, list]:
        """Find best matching response and return alternatives."""
        ranked_responses = self._rank_responses_by_similarity(predicted_24d)
        closest_response, closest_similarity = ranked_responses[0]
        alternatives = [response for response, _ in ranked_responses[1:3]]
        return closest_response, closest_similarity, alternatives

    def _embed_question_to_384d(self, question_text: str) -> np.ndarray:
        """Convert question text to 384-dimensional embedding."""
        embedding = self.embedder.encode([question_text], convert_to_numpy=True)[0]
        return embedding

    def _project_to_latent_space(self, embedding_384d: np.ndarray) -> np.ndarray:
        """Project embedding through PCA to 24D latent space."""
        latent_vector = self.pca.transform([embedding_384d])[0]
        return latent_vector

    def _predict_response_vector(self, question_latent: np.ndarray) -> np.ndarray:
        """Use predictor network to generate persona-specific response vector."""
        response_vector = self.predictor.predict(question_latent)
        return response_vector

    def _match_to_response_option(
        self, response_vector: np.ndarray
    ) -> Tuple[str, float, list]:
        """Find the best response option for the predicted vector."""
        best_response, similarity, alternatives = self._find_best_response(
            response_vector
        )
        return best_response, similarity, alternatives

    def consult(self, question_text: str) -> Dict:
        """Consult the oracle and return a response."""
        # Pipeline: embed → project → predict → match
        question_embedding = self._embed_question_to_384d(question_text)
        question_latent = self._project_to_latent_space(question_embedding)
        response_vector = self._predict_response_vector(question_latent)
        best_response, similarity, alternatives = self._match_to_response_option(
            response_vector
        )

        return {
            "question": question_text,
            "response": best_response,
            "similarity": similarity,
            "alternatives": alternatives,
            "persona": self.persona_name,
        }


if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("MAGIC 8 BALL - PCA + Predictor Edition")
    print("=" * 70)
    print()

    # List available personas
    models_dir = Path("toys_models")
    if not models_dir.exists():
        print("Error: toys_models/ directory not found")
        print("Run: python -m toys.train_persona --text-path <text> --persona-name <name>")
        sys.exit(1)

    predictors = sorted(set(p.stem.replace("predictor_", "") for p in models_dir.glob("predictor_*.pt")))
    if not predictors:
        print("No trained personas found in toys_models/")
        sys.exit(1)

    print("Available personas:")
    for p in predictors:
        print(f"  • {p}")
    print()

    # Prompt for persona
    while True:
        persona = input("Choose a persona: ").strip().lower()
        if persona in predictors:
            break
        print(f"Invalid. Choose from: {', '.join(predictors)}")

    # Initialize
    print()
    ball = Magic8Ball(persona_name=persona)

    print("\n" + "=" * 70)
    print(f"CONSULTING {persona.upper()}'S MAGIC 8 BALL")
    print("=" * 70)
    print("Type questions. Type 'exit' to quit.\n")

    # Test queries
    test_queries = [
        "Should we deploy this immediately?",
        "This approach seems uncertain.",
        "This is risky and needs discussion.",
    ]

    print("=== Test Queries ===\n")
    for q in test_queries:
        result = ball.consult(q)
        print(f"Q: {q}")
        print(f"   🎱 A: {result['response']} (similarity: {result['similarity']:.4f})")
        if result['alternatives']:
            print(f"   Also: {', '.join(result['alternatives'][:2])}")
        print()

    # Interactive mode
    print("=== Interactive Mode ===\n")
    while True:
        try:
            user_q = input("You: ").strip()
            if not user_q:
                continue
            if user_q.lower() in ["exit", "quit", "q"]:
                print("\nThe oracle fades into silence.\n")
                break

            result = ball.consult(user_q)
            print(f"🎱 {result['response']}")
            if result['alternatives']:
                print(f"   (Also considering: {', '.join(result['alternatives'][:2])})")
            print()

        except KeyboardInterrupt:
            print("\n\nThe oracle fades into silence.\n")
            break
        except Exception as e:
            print(f"Error: {e}\n")
