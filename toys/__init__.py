"""
Toys - Lightweight decision oracles using PCA semantic projection + predictors.

Eleven literary personas:
- Velveteen Rabbit (love, becoming real)
- Frankenstein (obsession, creation)
- Jane Eyre (independence, passion)
- Don Quixote (idealism vs reality)
- The Waste Land (fragmentation, modernism)
- Mrs. Dalloway (consciousness, interiority)
- The Great Gatsby (dreams, illusion)
- Meditations (stoic acceptance, duty)
- Alice in Wonderland (logic, whimsy)
- Persuasion (social dynamics, restraint)
- Moby Dick (defiance, the sea)

Architecture:
1. Learn PCA from text (384D → 24D, Leech lattice)
2. Train predictor net (24D → 24D, learns semantic next-step)
3. Project questions through PCA
4. Pass through predictor
5. Match to responses/cards via cosine similarity

Result: Cheap, interpretable, persona-specific decision oracles.
"""

__version__ = "0.1.0"
