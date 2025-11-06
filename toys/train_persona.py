"""
Train Persona - Unified training pipeline for PCA + Predictor.

FLOW:
1. Read text file
2. Fit PCA: 384D → 24D
3. Project all chunks through PCA
4. Train predictor: 24D → 24D (chunk_i → chunk_i+1)
5. Save both models

RESULT:
- pca_{name}.pkl (PCA projection matrix)
- predictor_{name}.pt (24D → 24D neural net)
"""

import argparse
from pathlib import Path
import numpy as np
from toys.pca_trainer import PCATrainer
from toys.predictor_model import PredictorTrainer


def train_persona(text_path: str, persona_name: str, output_dir: str = "toys_models"):
    """
    Train PCA + Predictor for a text source.

    Args:
        text_path: Path to text file
        persona_name: Name for this persona (hallie, victor, etc)
        output_dir: Where to save models
    """
    # Validate text file exists and is readable
    text_path = Path(text_path)
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")

    try:
        with open(text_path, encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        # Fallback to ISO-8859-1 if UTF-8 fails
        print(f"⚠️  UTF-8 decoding failed for {text_path}, trying ISO-8859-1...")
        with open(text_path, encoding="iso-8859-1") as f:
            text = f.read()

    if not text or len(text.strip()) < 100:
        raise ValueError(
            f"Text file too small ({len(text)} chars). "
            "Need at least 100 characters for meaningful training."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"TRAINING PERSONA: {persona_name.upper()}")
    print(f"{'='*70}\n")

    # Step 1: Train PCA
    print("STEP 1: Fitting PCA (384D → 24D)")
    print("-" * 70)
    pca_trainer = PCATrainer(output_dim=24)
    pca_path = output_dir / f"pca_{persona_name}.pkl"
    pca_trainer.fit(text_path, str(pca_path))

    # Step 2: Get latent projections
    print("\nSTEP 2: Projecting chunks through PCA")
    print("-" * 70)
    with open(text_path) as f:
        text = f.read()
    chunks = pca_trainer._split_into_chunks(text)
    embeddings = pca_trainer.embedder.encode(chunks, convert_to_numpy=True)
    latent_chunks = pca_trainer.pca.transform(embeddings)
    print(f"Projected {len(chunks)} chunks to 24D")
    print(f"Latent shape: {latent_chunks.shape}")

    # Step 3: Train predictor
    print("\nSTEP 3: Training predictor (24D → 24D)")
    print("-" * 70)
    predictor_trainer = PredictorTrainer(latent_dim=24)
    print(f"🧠 Using device: {predictor_trainer.device.upper()}")
    predictor_trainer.fit(latent_chunks, epochs=100, batch_size=2, patience=15)
    predictor_path = output_dir / f"predictor_{persona_name}.pt"
    predictor_trainer.save(str(predictor_path))

    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"PCA model:       {pca_path}")
    print(f"Predictor model: {predictor_path}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PCA + Predictor persona models"
    )
    parser.add_argument("--text-path", required=True, help="Path to text file")
    parser.add_argument(
        "--persona-name", required=True, help="Name for this persona"
    )
    parser.add_argument(
        "--output-dir",
        default="toys_models",
        help="Where to save models (default: toys_models)",
    )

    args = parser.parse_args()
    train_persona(args.text_path, args.persona_name, args.output_dir)
