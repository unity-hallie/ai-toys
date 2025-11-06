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
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Load PCA
        pca_path = self.models_dir / f"pca_{persona_name}.pkl"
        if not pca_path.exists():
            raise FileNotFoundError(f"PCA model not found: {pca_path}")
        with open(pca_path, "rb") as f:
            self.pca = pickle.load(f)
        print(f"✅ Loaded PCA model: {pca_path}")

        # Load Predictor
        predictor_path = self.models_dir / f"predictor_{persona_name}.pt"
        if not predictor_path.exists():
            raise FileNotFoundError(f"Predictor not found: {predictor_path}")
        self.predictor = PredictorTrainer.load(str(predictor_path))
        print(f"✅ Loaded Predictor model: {predictor_path}")

        # Pre-embed responses through PCA
        print(f"📚 Embedding {len(self.RESPONSES)} response options...")
        response_embeddings_384d = self.embedder.encode(
            self.RESPONSES, convert_to_numpy=True
        )
        self.response_vectors_24d = self.pca.transform(response_embeddings_384d)
        print(f"   Shape: {self.response_vectors_24d.shape}")

    def _find_best_response(
        self, predicted_24d: np.ndarray
    ) -> Tuple[str, float, list]:
        """Match predicted response to the 10 options.

        THE PROBLEM:
        ============

        You have:
        - predicted_24d: the network's prediction (24 numbers)
        - 10 response options, each also 24 numbers

        GOAL: Which of the 10 is closest to the prediction?

        THE SOLUTION: Cosine Similarity
        ===============================

        Imagine two arrows in space:
        - One points in the direction of predicted_24d
        - One points in the direction of response_i

        Cosine similarity measures the angle between them:
        - Angle = 0°   (same direction)     → similarity = 1.0  (perfect match)
        - Angle = 90°  (perpendicular)      → similarity = 0.0  (unrelated)
        - Angle = 180° (opposite direction) → similarity = -1.0 (inverse)

        FORMULA (math-free version):
        ============================
        similarity = (predicted · response) / (length_pred × length_resp)

        Where:
        - (predicted · response) = how much they point in same direction
        - length = how far the arrow is from origin
        - Division = normalize so result is between -1 and 1

        WHY THIS METRIC?
        ================
        - Fast to compute
        - Makes sense geometrically
        - Works well for high-dimensional spaces
        - Robust to magnitude (only direction matters)

        PRACTICAL EXAMPLE:
        ==================
        If predicted_24d is "cautious + uncertain + wait"
        And response "Yes, but proceed cautiously" is also "cautious + uncertain + yes"
        Then similarity is high (0.35-0.55) because they point similar directions.

        REAL ALGORITHM:
        ===============
        1. For each of 10 responses:
           a. Calculate cosine similarity
        2. Sort by similarity (highest first)
        3. Return: best match, its score, and 2 alternatives
        """
        similarities = {}

        for i, response_text in enumerate(self.RESPONSES):
            response_24d = self.response_vectors_24d[i]

            # Cosine similarity: (dot product) / (norms)
            dot_product = np.dot(predicted_24d, response_24d)
            pred_norm = np.linalg.norm(predicted_24d)
            resp_norm = np.linalg.norm(response_24d)

            # Epsilon (1e-8) prevents division by zero
            similarity = dot_product / (pred_norm * resp_norm + 1e-8)
            similarities[response_text] = similarity

        # Sort by similarity (highest first)
        sorted_responses = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        closest_response, closest_similarity = sorted_responses[0]
        alternatives = [r for r, _ in sorted_responses[1:3]]

        return closest_response, closest_similarity, alternatives

    def consult(self, question_text: str) -> Dict:
        """
        Consult the magic 8 ball.

        Args:
            question_text: The question/proposal

        Returns:
            Dict with response, similarity, alternatives, persona

        ❓ INFERENCE QUESTIONS:

        1. The pipeline: embed → PCA project → predictor → match
           - Is this the right order?
           - Could we skip the predictor and just match question directly?
           - Pro: Simpler, fewer moving parts
           - Con: Loses persona-specific "next-step" information

        2. Should we scale the predictor output?
           - Currently: use as-is
           - Alternative: normalize to unit norm before matching
           - Does this change the rankings?

        3. Cosine similarity has a 1e-8 epsilon. Why?
           - To prevent division by zero?
           - Should we use torch.nn.functional.cosine_similarity instead?
           - Pro: More numerically stable
           - Con: Different implementation, different results

        4. Should we return confidence scores?
           - Currently: similarity is [0..1] range
           - Alternative: convert to probability (softmax)?
           - Alternative: confidence = (max - second_max) / max?
           - Pro: More informative about decision confidence

        5. How many alternatives should we return?
           - Currently: hardcoded top 3
           - Should this be configurable?
           - Should we only return if gap is small?

        6. Should we cache the predicted response?
           - Currently: compute fresh each time
           - For same question asked twice: redundant computation?
           - Con: Memory overhead, invalidation issues

        7. Should question preprocessing happen?
           - Currently: raw text
           - Alternative: lowercase, remove punctuation, normalize
           - Does normalization help or hurt persona-specific encoding?
        """
        # Embed question to 384D
        question_384d = self.embedder.encode([question_text], convert_to_numpy=True)[0]

        # Project through PCA to 24D
        question_24d = self.pca.transform([question_384d])[0]

        # Predict response through predictor
        predicted_response_24d = self.predictor.predict(question_24d)

        # Find the closest response
        closest_response, closest_similarity, alternatives = self._find_best_response(
            predicted_response_24d
        )

        return {
            "question": question_text,
            "response": closest_response,
            "similarity": closest_similarity,
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
