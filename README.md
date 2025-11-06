# Toys: PCA + Predictor Decision Oracles

Simple, lightweight decision oracles using PCA semantic projection + cheap toy predictors.

## Architecture

Each persona (text source) learns:

1. **PCA Model**: 384D (sentence-transformers) → 24D (Leech lattice dimension)
2. **Predictor**: 24D → 24D neural network (~2400 parameters)
   - Trained to predict chunk_i+1 from chunk_i in latent space
   - Cheap, fast, interpretable

## The Twelve Personas

**Romantic/Gothic:**
- **Velveteen Rabbit** (Margery Williams) - Love, becoming real, tenderness
- **Frankenstein** (Mary Shelley) - Creation, obsession, suffering, ambition
- **Jane Eyre** (Charlotte Brontë) - Independence, passion, Gothic atmosphere
- **Don Quixote** (Cervantes) - Idealism vs reality, adventure, madness

**Modernist/Introspective:**
- **The Waste Land** (T.S. Eliot) - Fragmentation, modernism, decay
- **Mrs. Dalloway** (Virginia Woolf) - Consciousness, interiority, time-flux
- **The Great Gatsby** (F. Scott Fitzgerald) - Dreams, illusion, excess

**Philosophical/Political:**
- **Meditations** (Marcus Aurelius) - Stoic acceptance, duty, inner peace
- **Das Kapital** (Karl Marx) - Dialectics, labor, historical materialism, revolution
- **Alice in Wonderland** (Lewis Carroll) - Logic, whimsy, nonsense
- **Persuasion** (Jane Austen) - Social dynamics, restraint, delayed satisfaction
- **Moby Dick** (Herman Melville) - Obsession, the sea, defiance

## Quick Start

### 1. Install Dependencies

```bash
pip install -e .
```

### 2. Download Texts

```bash
python -m toys.download_texts
```

This downloads from Project Gutenberg into `data/`.

### 3. Train All Personas

```bash
# All at once (recommended):
python toys/setup_all_personas.py

# Or one at a time:
python -m toys.train_persona --text-path data/velveteen_rabbit.txt --persona-name velveteen
python -m toys.train_persona --text-path data/frankenstein.txt --persona-name frankenstein
python -m toys.train_persona --text-path data/jane_eyre.txt --persona-name jane
python -m toys.train_persona --text-path data/don_quixote.txt --persona-name quixote
python -m toys.train_persona --text-path data/waste_land.txt --persona-name waste_land
python -m toys.train_persona --text-path data/mrs_dalloway.txt --persona-name woolf
python -m toys.train_persona --text-path data/great_gatsby.txt --persona-name gatsby
python -m toys.train_persona --text-path data/meditations.txt --persona-name marcus
python -m toys.train_persona --text-path data/alice_wonderland.txt --persona-name alice
python -m toys.train_persona --text-path data/das_kapital.txt --persona-name marx
python -m toys.train_persona --text-path data/persuasion.txt --persona-name persuasion
python -m toys.train_persona --text-path data/moby_dick.txt --persona-name moby
```

This trains PCA + Predictor for each text source.
Results: `toys_models/pca_*.pkl` and `toys_models/predictor_*.pt`

### 4. Run Magic 8 Ball

```bash
python -m toys.magic_8_ball
```

Choose persona, then ask questions. Example:
```
Q: Should we deploy immediately?
🎱 A: Yes, but proceed cautiously

Q: This approach is risky
🎱 A: Too risky
```

### 5. Run Tarot

```bash
python -m toys.tarot
```

Choose persona, then:
```
single <question>     # Draw one card
spread <question>     # Draw 10-card spread
```

## How It Works

### Inference Flow

```
Question
   ↓
Embed to 384D (sentence-transformers)
   ↓
Project through PCA → 24D
   ↓
Pass through Predictor → 24D response
   ↓
Compute similarity to responses/cards
   ↓
Return closest match
```

### Why 24D?

The Leech lattice (24D) has exceptional symmetry properties in the kissing number and automorphism group. Interesting for potential emergent structure in semantic projection.

### Why Predictor?

Instead of treating the latent vector as a static embedding, the predictor learns what that persona's semantic "next step" would be. This captures directional tendency in their thought space.

## Files

```
toys/
├── __init__.py
├── pca_trainer.py        # Fit PCA on texts
├── predictor_model.py    # 24D → 24D neural net
├── train_persona.py      # Unified training pipeline
├── magic_8_ball.py       # Decision oracle
├── tarot.py              # Card oracle
├── download_texts.py     # Fetch from Project Gutenberg
└── setup_all_personas.py # Batch training script

toys_models/
├── pca_velveteen.pkl         ├── predictor_velveteen.pt
├── pca_frankenstein.pkl      ├── predictor_frankenstein.pt
├── pca_jane.pkl              ├── predictor_jane.pt
├── pca_quixote.pkl           ├── predictor_quixote.pt
├── pca_waste_land.pkl        ├── predictor_waste_land.pt
├── pca_woolf.pkl             ├── predictor_woolf.pt
├── pca_gatsby.pkl            ├── predictor_gatsby.pt
├── pca_marcus.pkl            ├── predictor_marcus.pt
├── pca_marx.pkl              ├── predictor_marx.pt
├── pca_alice.pkl             ├── predictor_alice.pt
├── pca_persuasion.pkl        ├── predictor_persuasion.pt
└── pca_moby.pkl              └── predictor_moby.pt

data/
├── velveteen_rabbit.txt
├── frankenstein.txt
├── jane_eyre.txt
├── don_quixote.txt
├── waste_land.txt
├── mrs_dalloway.txt
├── great_gatsby.txt
├── meditations.txt
├── das_kapital.txt
├── alice_wonderland.txt
├── persuasion.txt
└── moby_dick.txt
```

## Why This Approach?

vs. Response Encoder:
- **Response Encoder** treats encoder output as a generic latent. Prone to averaging.
- **Predictor** learns text-specific "next step" semantics. Directional. Interpretable.

vs. Persona Encoder:
- **Persona Encoder** uses bottleneck to learn shared structure across personas. More expressive.
- **Predictor** is cheap, fast, and directly optimized for the task (sequence prediction).

vs. No Model:
- **No Model** just cosine similarity of raw embeddings. Loses learned structure.
- **Predictor** learns semantic direction specific to that text.

## Development

Run tests:
```bash
pytest tests/
```

Check models:
```bash
ls -lh toys_models/
```

Inspect a trained PCA:
```python
import pickle
with open("toys_models/pca_velveteen.pkl", "rb") as f:
    pca = pickle.load(f)
print(f"Explained variance: {pca.explained_variance_ratio_}")
```

## Philosophy

"Toys" because:
- They're deliberately simple (not production ML)
- They're playful (magic 8 ball + tarot)
- They show the mechanism clearly (PCA + small net)
- They work with ~2400 parameters per persona

No hidden layers of abstractions. What you see is what you get.
