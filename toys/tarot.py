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
from toys.magic_8_ball import get_embedder

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

# Minor Arcana - Wands (14 cards)
WANDS = [
    "Ace of Wands",
    "Two of Wands",
    "Three of Wands",
    "Four of Wands",
    "Five of Wands",
    "Six of Wands",
    "Seven of Wands",
    "Eight of Wands",
    "Nine of Wands",
    "Ten of Wands",
    "Page of Wands",
    "Knight of Wands",
    "Queen of Wands",
    "King of Wands",
]

# Minor Arcana - Cups (14 cards)
CUPS = [
    "Ace of Cups",
    "Two of Cups",
    "Three of Cups",
    "Four of Cups",
    "Five of Cups",
    "Six of Cups",
    "Seven of Cups",
    "Eight of Cups",
    "Nine of Cups",
    "Ten of Cups",
    "Page of Cups",
    "Knight of Cups",
    "Queen of Cups",
    "King of Cups",
]

# Minor Arcana - Swords (14 cards)
SWORDS = [
    "Ace of Swords",
    "Two of Swords",
    "Three of Swords",
    "Four of Swords",
    "Five of Swords",
    "Six of Swords",
    "Seven of Swords",
    "Eight of Swords",
    "Nine of Swords",
    "Ten of Swords",
    "Page of Swords",
    "Knight of Swords",
    "Queen of Swords",
    "King of Swords",
]

# Minor Arcana - Pentacles (14 cards)
PENTACLES = [
    "Ace of Pentacles",
    "Two of Pentacles",
    "Three of Pentacles",
    "Four of Pentacles",
    "Five of Pentacles",
    "Six of Pentacles",
    "Seven of Pentacles",
    "Eight of Pentacles",
    "Nine of Pentacles",
    "Ten of Pentacles",
    "Page of Pentacles",
    "Knight of Pentacles",
    "Queen of Pentacles",
    "King of Pentacles",
]

# All tarot cards (78 total)
ALL_TAROT_CARDS = MAJOR_ARCANA + WANDS + CUPS + SWORDS + PENTACLES


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
        self.embedder = get_embedder("all-MiniLM-L6-v2")

        self.pca = self._load_pca_model(persona_name)
        self.predictor = self._load_predictor_model(persona_name)
        self._validate_model_compatibility()
        self.card_vectors_24d = self._precompute_card_embeddings()

    def _load_pca_model(self, persona_name: str):
        """Load and validate PCA model from disk."""
        pca_path = self.models_dir / f"pca_{persona_name}.pkl"
        if not pca_path.exists():
            raise FileNotFoundError(f"PCA model not found: {pca_path}")
        try:
            with open(pca_path, "rb") as f:
                pca = pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"Corrupted PCA model file {pca_path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load PCA model {pca_path}: {e}")
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

    def _precompute_card_embeddings(self) -> np.ndarray:
        """Precompute all tarot card vectors in latent space."""
        print(f"📚 Embedding {len(ALL_TAROT_CARDS)} tarot cards...")
        card_embeddings_384d = self.embedder.encode(
            ALL_TAROT_CARDS, convert_to_numpy=True
        )
        card_vectors_24d = self.pca.transform(card_embeddings_384d)
        print(f"   Shape: {card_vectors_24d.shape}")
        return card_vectors_24d

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

        ❓ TAROT DESIGN QUESTIONS:

        1. Temporal context: why include time?
           - Currently: add hour and day-of-week to question
           - Intuition: when you ask matters
           - But: is this scientifically justified?
           - Alternative: don't include time
           - How much does this actually change results?

        2. Distance metric: Euclidean vs. Cosine?
           - Currently: Euclidean distance
           - Magic 8 ball uses: cosine similarity
           - Which is better for card selection?
           - Does the choice matter much?

        3. Should draws be without replacement?
           - Currently: can draw the same card twice
           - Alternative: remove card after drawing
           - Pro: More variety in spread
           - Con: Changes semantics of "distance"

        4. Should we use major arcana only?
           - Currently: yes, 22 cards
           - Alternative: full deck (78), or minors only (56)
           - What spreads need what deck size?

        5. Default num_cards = 10. Why?
           - Standard tarot spread sizes: 1, 3, 10, 21
           - Is 10 arbitrary or meaningful?
           - Should we support common spread types?

        6. The temporal context is semantic embedding.
           - Does sentence_transformers handle time well?
           - Alternative: separate temporal encoding + concatenation?
           - Should time be a hard constraint (e.g., only same-hour results)?

        7. Should we support minor arcana?
           - Currently: only majors
           - Would require a larger trained model
           - Or: train separate predictor for minors?
           - What does that add?
        """
        # Embed question with temporal context to 384D
        question_384d = self._embed_question_with_time(question_text)

        # Project through PCA to 24D
        question_24d = self.pca.transform([question_384d])[0]

        # Predict response through predictor
        predicted_response_24d = self.predictor.predict(question_24d)

        # Find closest cards
        drawn = self._find_closest_cards(predicted_response_24d, num_cards)
        return drawn

    def _find_closest_cards(self, predicted_24d: np.ndarray, num_cards: int) -> List[str]:
        """Find the closest tarot cards to the prediction using Euclidean distance."""
        distances = {}

        for i, card_name in enumerate(ALL_TAROT_CARDS):
            card_24d = self.card_vectors_24d[i]
            # Euclidean distance: sqrt(sum of squared differences)
            difference = predicted_24d - card_24d
            distance = np.linalg.norm(difference)
            distances[card_name] = distance

        # Sort by distance (closest = smallest)
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
