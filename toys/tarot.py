"""
Tarot - Card oracle using PCA + Predictor models.

ARCHITECTURE:
=============
1. Load PCA model for persona (384D → 24D projection)
2. Load Predictor model for persona (24D → 24D response)
3. Embed question to 384D, add temporal context
4. Project through PCA to 24D
5. Pass through predictor to get predicted response
6. Match to 22 major arcana cards (also encoded through PCA)
7. Return closest cards

DECK:
=====
22 Major Arcana cards (standard Tarot)
"""

import pickle
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from toys.predictor_model import PredictorTrainer
import logging

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


# Major Arcana (22 cards)
MAJOR_ARCANA = [
    "The Fool",
    "The Magician",
    "The High Priestess",
    "The Empress",
    "The Emperor",
    "The Hierophant",
    "The Lovers",
    "The Chariot",
    "Strength",
    "The Hermit",
    "Wheel of Fortune",
    "Justice",
    "The Hanged Man",
    "Death",
    "Temperance",
    "The Devil",
    "The Tower",
    "The Star",
    "The Moon",
    "The Sun",
    "Judgement",
    "The World",
]


class TarotReader:
    """Card oracle using PCA + Predictor models."""

    def __init__(self, persona_name: str, models_dir: str = "toys_models"):
        """
        Initialize tarot reader with persona models.

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

        # Pre-embed cards through PCA
        print(f"📚 Embedding {len(MAJOR_ARCANA)} major arcana cards...")
        card_embeddings_384d = self.embedder.encode(
            MAJOR_ARCANA, convert_to_numpy=True
        )
        self.card_vectors_24d = self.pca.transform(card_embeddings_384d)
        print(f"   Shape: {self.card_vectors_24d.shape}")

    def _embed_question_with_time(self, question_text: str) -> np.ndarray:
        """
        Embed question with temporal context.

        The idea: when you ask a question matters.
        Include hour, day-of-week as context.
        """
        now = datetime.now()
        temporal_context = f"[{now.strftime('%A %I%p')}] {question_text}"
        embedding = self.embedder.encode(
            [temporal_context], convert_to_numpy=True
        )[0]
        return embedding

    def single_card(self, question_text: str) -> str:
        """Draw a single card for the question."""
        return self.draw_cards(question_text, num_cards=1)[0]

    def draw_cards(self, question_text: str, num_cards: int = 10) -> List[str]:
        """
        Draw cards for the question.

        Args:
            question_text: The question/query
            num_cards: How many cards to draw

        Returns:
            List of card names
        """
        # Embed question with temporal context to 384D
        question_384d = self._embed_question_with_time(question_text)

        # Project through PCA to 24D
        question_24d = self.pca.transform([question_384d])[0]

        # Predict response through predictor
        predicted_response_24d = self.predictor.predict(question_24d)

        # Compute distance to each card
        distances = {}
        for i, card_name in enumerate(MAJOR_ARCANA):
            card_24d = self.card_vectors_24d[i]
            # Euclidean distance
            distance = np.linalg.norm(predicted_response_24d - card_24d)
            distances[card_name] = distance

        # Sort by distance (closest = best match)
        sorted_cards = sorted(distances.items(), key=lambda x: x[1])
        drawn = [card for card, _ in sorted_cards[:num_cards]]

        return drawn


if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("TAROT READER - PCA + Predictor Edition")
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
    reader = TarotReader(persona_name=persona)

    print("\n" + "=" * 70)
    print(f"CONSULTING {persona.upper()}'S TAROT")
    print("=" * 70)
    print()

    # Test queries
    test_queries = [
        "What should I focus on?",
        "Is this the right path?",
        "What do I need to understand?",
    ]

    print("=== Single Card Draws ===\n")
    for q in test_queries:
        card = reader.single_card(q)
        print(f"Q: {q}")
        print(f"   🃏 {card}\n")

    # Interactive mode
    print("=== Interactive Mode ===")
    print("Commands: 'single <question>' or 'spread <question>' or 'exit'\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nThe cards return to the deck.\n")
                break

            # Parse command
            if user_input.lower().startswith("single "):
                question = user_input[7:].strip()
                card = reader.single_card(question)
                print(f"🃏 {card}\n")

            elif user_input.lower().startswith("spread "):
                question = user_input[7:].strip()
                cards = reader.draw_cards(question, num_cards=10)
                print("\n10-Card Spread:")
                for i, card in enumerate(cards, 1):
                    print(f"  {i:2}. {card}")
                print()

            else:
                # Default: single card for the query
                card = reader.single_card(user_input)
                print(f"🃏 {card}\n")

        except KeyboardInterrupt:
            print("\n\nThe cards return to the deck.\n")
            break
        except Exception as e:
            print(f"Error: {e}\n")
