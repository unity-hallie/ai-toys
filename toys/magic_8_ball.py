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
from typing import Dict
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

    def consult(self, question_text: str) -> Dict:
        """
        Consult the magic 8 ball.

        Args:
            question_text: The question/proposal

        Returns:
            Dict with response, similarity, alternatives, persona
        """
        # Embed question to 384D
        question_384d = self.embedder.encode([question_text], convert_to_numpy=True)[0]

        # Project through PCA to 24D
        question_24d = self.pca.transform([question_384d])[0]

        # Predict response through predictor
        predicted_response_24d = self.predictor.predict(question_24d)

        # Compute cosine similarity to each response
        similarities = {}
        for i, response_text in enumerate(self.RESPONSES):
            response_24d = self.response_vectors_24d[i]
            # Cosine similarity
            sim = np.dot(predicted_response_24d, response_24d) / (
                np.linalg.norm(predicted_response_24d)
                * np.linalg.norm(response_24d)
                + 1e-8
            )
            similarities[response_text] = sim

        # Sort by similarity
        sorted_responses = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        closest_response, closest_similarity = sorted_responses[0]
        alternatives = [r for r, _ in sorted_responses[1:3]]

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
