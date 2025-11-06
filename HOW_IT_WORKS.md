# How Toys Works

This guide explains the pipeline and key concepts.

## The Big Picture

You have books. You want decision oracles that sound like each book's voice.

**Pipeline:**

```
Book (text)
    ↓
Split into chunks
    ↓
Embed to numbers (SentenceTransformers)
    ↓
Compress numbers (PCA)
    ↓
Teach network the book's patterns (Predictor)
    ↓
Ask questions → Get answers in that voice
```

Each step is explained below.

---

## Step 1: Embedding (Text → Numbers)

**What's happening:**

Text is converted to 384-dimensional vectors using SentenceTransformers, a pre-trained model:
```
"Should I wait?" → [0.23, -0.51, 0.12, 0.44, ..., 0.89]  (384 numbers)
```

SentenceTransformers learned from billions of sentences to map semantically similar phrases to nearby vectors. The 384 dimensions capture various aspects of meaning, though they're not individually interpretable. Together, they form a dense representation of semantic content.

---

## Step 2: Chunking (Book → Pieces)

**What's happening:**

The text is split into ~100-word chunks. Each chunk is independently embedded into a 384-dimensional vector:

```
"It was the best of times, it was the worst of times, ..."
    ↓
Chunk 1: "It was the best of times, it was the worst of times"
Chunk 2: "It was the age of wisdom, it was the age of foolishness"
    ↓
Each becomes a 384D vector
```

Chunking at sentence boundaries preserves semantic coherence while keeping individual vectors tractable.

---

## Step 3: PCA (Dimensionality Reduction)

**The Problem:**

384 dimensions is high-dimensional and computationally expensive for training.

**The Solution:**

PCA (Principal Component Analysis) projects the 384D vectors to 24D by keeping the principal components that explain 90%+ of variance. This is a standard dimensionality reduction technique that balances expressiveness with computational efficiency.

**Why 24D?**

24 dimensions provides a good balance between expressiveness and computational efficiency. This specific dimensionality allows for future work exploring structural symmetries in semantic space.

---

## Step 4: Training the Predictor Network

**The Goal:**

Train a neural network to predict the next chunk's embedding given the current chunk. This forces the network to learn the book's semantic patterns and progression.

**Training Setup:**

```
For each training example:
  Input:  chunk_i vector (24D)
  Target: chunk_{i+1} vector (24D)

The network minimizes prediction error over all consecutive pairs.
```

This is a sequence-to-sequence learning task on the embedding space.

**Architecture:**

```
24D input
    ↓
48D hidden layer (ReLU activation)
    ↓
24D hidden layer (ReLU activation)
    ↓
24D output (predicted next embedding)
```

The network has ~2,400 parameters. A small architecture on limited data reduces overfitting and forces the network to learn general semantic patterns rather than memorizing.

**Training:**

- Epochs: up to 100 (early stopping if validation loss doesn't improve for 15 epochs)
- Batch size: 4
- Optimizer: standard SGD with gradient descent

---

## Step 5: Answering Questions

**Magic 8 Ball:**

1. Embed the question: "Should I wait?" → 384D vector
2. Compress via PCA: → 24D vector
3. Pass through the trained network: → 24D predicted response vector
4. Find the closest match among 10 response embeddings using cosine similarity
5. Return that response

**Tarot:**

Same pipeline, but:
- Compares against 22 major arcana card embeddings (instead of 10 responses)
- Uses Euclidean distance for similarity (instead of cosine)
- Returns the card(s) with minimum distance

---

## The Math (Optional)

If you're curious about formulas:

### Cosine Similarity

```
similarity = (A · B) / (|A| × |B|)

Where:
- A · B = sum of (a_i × b_i) for all 24 dimensions
- |A| = sqrt(a_1² + a_2² + ... + a_24²)
```

Intuitively: measure the angle between two vectors.
- 0°: parallel (similarity = 1.0)
- 90°: perpendicular (similarity = 0.0)
- 180°: opposite (similarity = -1.0)

### Euclidean Distance

```
distance = sqrt((a_1 - b_1)² + (a_2 - b_2)² + ... + (a_24 - b_24)²)
```

Intuitively: how far apart are two points in 24D space?
- Same point: distance = 0
- Far apart: distance = large number

### ReLU

```
f(x) = max(0, x)

If x > 0: output = x
If x ≤ 0: output = 0
```

This simple bend lets neural networks learn nonlinear patterns.

---

## Key Design Decisions

1. **Pre-trained embeddings**: SentenceTransformers captures semantic relationships learned from billions of sentences. This avoids training embeddings from scratch.

2. **Dimensionality reduction via PCA**: Reduces 384 → 24 dimensions while retaining 90%+ of variance. Trades a small amount of expressiveness for significant computational gains.

3. **Sequence prediction as training signal**: Learning chunk_i → chunk_i+1 captures the statistical patterns of the text without explicit semantic annotation.

4. **Small network architecture**: ~2,400 parameters prevents overfitting on limited training data and encourages learning generalizable patterns.

5. **Similarity in reduced space**: Both cosine similarity (angle) and Euclidean distance (point distance) work as matching metrics in the PCA-projected space.

---

## Common Questions

**Q: Is this machine learning?**

A: Yes. It combines embeddings, dimensionality reduction, and neural networks.

**Q: Why does it work?**

A: Text has statistical structure—similar concepts tend to appear near each other. The network learns this structure via sequence prediction.

**Q: Is this actually learning the book's voice?**

A: To a degree. The network learns statistical patterns in how concepts flow within the text. Whether that constitutes "voice" is subjective.

**Q: Can I train on my own text?**

A: Yes:

```bash
python -m toys.train_persona --text-path mytext.txt --persona-name myvoice
python -m toys.magic_8_ball
```

**Q: Why 24D?**

A: It's a practical choice that balances model capacity with computational efficiency. Future work may explore whether semantic space exhibits lattice-like symmetries at this dimensionality.

---

## Implementation Details

The codebase has detailed docstrings. Key files:

- `toys/pca_trainer.py` - PCA dimensionality reduction
- `toys/predictor_model.py` - Neural network training and inference
- `toys/magic_8_ball.py` - Question-to-response matching via similarity
- `toys/tarot.py` - Card selection using distance metrics
