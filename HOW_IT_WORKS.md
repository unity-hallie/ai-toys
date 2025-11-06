# How Toys Works: A Gentle Explanation

This guide explains the core concepts without math trauma.

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

You have text. Computers can't think with words; they think with numbers.

SentenceTransformers converts:
```
"Should I wait?" → [0.23, -0.51, 0.12, 0.44, ..., 0.89]  (384 numbers)
```

**How does it work?**

SentenceTransformers is pre-trained on billions of sentences. It learned:
- Words with similar meaning get similar number patterns
- "wait" and "pause" are close together
- "wait" and "run" are far apart

**Why 384 numbers?**

That's just what the model outputs. Think of it as 384 different "aspects" of meaning:
- Aspect 1: Is this positive or negative? (value: 0.5 = slightly positive)
- Aspect 2: Is this about time? (value: 0.8 = very time-focused)
- Aspect 3: Is this about action? (value: -0.2 = not very action-focused)
- ... (381 more aspects)

These aspects aren't human-readable. But together they capture the meaning.

---

## Step 2: Chunking (Book → Pieces)

**What's happening:**

Your book is thousands of sentences. That's too much. We split it into roughly 100-word chunks.

```
"It was the best of times, it was the worst of times, ..."
    ↓ (split at periods, group by word count)
Chunk 1: "It was the best of times, it was the worst of times"
Chunk 2: "It was the age of wisdom, it was the age of foolishness"
...
```

Each chunk becomes a 384-number vector (embedding).

---

## Step 3: PCA (Compress Numbers)

**The Problem:**

384 numbers per chunk is a lot. Storage, computation, training time.

**The Solution: PCA**

PCA finds the 24 most important "directions" in the number space.

**Analogy:**

Imagine a landscape photo:
- Real world: infinite detail
- Camera: captures essence in 2D

PCA is like asking:
- What 24 axes would capture the most variation in this data?
- Like: "vertical" is important, "horizontal" is important, "brightness" is important
- But "pixel 237" probably isn't important

**Result:**

384 numbers → 24 numbers (still keeping 90%+ of variation)

**Why 24D?**

The Leech lattice is a special 24-dimensional geometric structure. Maybe semantic space wants to be that efficient too? (It's a guess, but it works.)

---

## Step 4: Teaching the Network (Predictor)

**The Goal:**

We want a network that learns: "Given this chunk, what would come next?"

This teaches it the book's semantic rhythm.

**How it learns:**

```
Training example:
Input:  chunk_0 vector (24D)
Output: chunk_1 vector (24D)  ← target

Network tries: prediction_0 = network(chunk_0)
Error: prediction_0 ≠ chunk_1
Learn: adjust weights to reduce error
```

Do this for thousands of examples. The network learns patterns:
- In Austen: discussions of society → considerations of propriety
- In Melville: obsession → more obsession (gets intense)
- In Woolf: consciousness → fragmentation → consciousness

**The Network Architecture:**

```
24D input
    ↓
Expand to 48D (think about more aspects)
    ↓
ReLU (introduce nonlinearity: f(x) = max(0, x))
    ↓
Compress back to 24D
    ↓
ReLU again
    ↓
Output 24D (final prediction)
```

Why this shape?
- Expand: lets the network think creatively
- ReLU: bends linearity so it can learn curved patterns
- Compress: synthesize back to core concepts
- Small size: (2,400 parameters) avoids memorization

**Training process:**

```
For 100 epochs (or until improvement stops):
  For each mini-batch of 4 examples:
    1. Predict: guess what comes next
    2. Compute error: how wrong was I?
    3. Learn: adjust weights to be less wrong
```

Early stopping: if the error stops improving for 15 epochs, we're done.

---

## Step 5: Answering Questions

**Magic 8 Ball:**

1. Embed question: "Should I wait?" → 384D vector
2. Compress through PCA: → 24D vector
3. Predict through network: → 24D "semantic response"
4. Compare to 10 response options (also 24D)
5. Find closest using cosine similarity (arrow angle)
6. Return: "Revisit later"

**Tarot:**

Same as above, but:
- Compare to 22 major arcana cards (not 10 responses)
- Use Euclidean distance (how close, not arrow angle)
- Return: closest card(s)

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

## Key Insights

1. **Embeddings are learned patterns**: SentenceTransformers distilled 1 billion sentences into number recipes.

2. **PCA is compression**: Find the essential axes, discard noise.

3. **Sequence prediction teaches style**: If you learn chunk_i → chunk_i+1, you learn the book's semantic rhythm.

4. **Small networks are better**: When training on small data, a small network is forced to learn patterns (not memorize).

5. **Similarity/Distance are geometric**: In high-dimensional space, you can measure angles and distances like in 2D.

---

## Common Questions

**Q: Is this machine learning?**

A: Yes. It uses neural networks, embeddings, and dimensionality reduction. But the concepts are simpler than most deep learning.

**Q: Why does it work?**

A: Because text naturally has semantic structure. Similar ideas tend to follow each other. The network just learns those patterns.

**Q: Is this sentient?**

A: No. It's learned to complete patterns. It doesn't understand meaning the way you do.

**Q: Can I train on my own texts?**

A: Yes! Run:

```bash
python -m toys.train_persona --text-path mytext.txt --persona-name myvoice
python -m toys.magic_8_ball
```

**Q: Why Leech lattice?**

A: It's the densest sphere packing in 24D. Maybe semantic space wants to be that efficient. (It's a hunch, not proven.)

---

## For the Mathematically Curious

All the math is in the docstrings. Each method has:
1. Plain-language explanation
2. Intuitive analogy
3. The actual formula (if interested)

Look for docstrings marked with:
- "WHAT'S HAPPENING:" (start here)
- "THE MATH (if you're curious):" (optional)

---

## Files to Read

- `toys/pca_trainer.py` - How to compress 384D → 24D
- `toys/predictor_model.py` - How the network learns
- `toys/magic_8_ball.py` - How to match questions to responses
- `toys/tarot.py` - How to find semantic cards

All have detailed docstrings explaining the "why" as well as the "what".
