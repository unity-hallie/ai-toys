"""
Setup All Personas - Train PCA + Predictor for all texts in one go.

Run this after downloading texts:
  python toys/setup_all_personas.py
"""

from pathlib import Path
from toys.train_persona import train_persona


if __name__ == "__main__":
    personas = [
        ("data/velveteen_rabbit.txt", "velveteen"),
        ("data/frankenstein.txt", "frankenstein"),
        ("data/waste_land.txt", "waste_land"),
        ("data/persuasion.txt", "persuasion"),
        ("data/moby_dick.txt", "moby"),
        ("data/mrs_dalloway.txt", "woolf"),
        ("data/great_gatsby.txt", "gatsby"),
        ("data/jane_eyre.txt", "jane"),
        ("data/don_quixote.txt", "quixote"),
        ("data/alice_wonderland.txt", "alice"),
        ("data/meditations.txt", "marcus"),
        ("data/das_kapital.txt", "marx"),
    ]

    print("\n" + "=" * 70)
    print("TRAINING ALL PERSONAS")
    print("=" * 70)

    for text_path, persona_name in personas:
        if not Path(text_path).exists():
            print(f"\n⚠️  {text_path} not found. Run: python -m toys.download_texts")
            continue

        train_persona(text_path, persona_name, output_dir="toys_models")

    print("\n" + "=" * 70)
    print("✅ ALL PERSONAS TRAINED")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  Magic 8 Ball: python -m toys.magic_8_ball")
    print("  Tarot:        python -m toys.tarot")
    print()
