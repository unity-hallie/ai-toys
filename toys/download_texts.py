"""
Download texts from Project Gutenberg.

The four personas:
- Velveteen Rabbit (Margery Williams)
- Frankenstein (Mary Shelley)
- The Waste Land (T.S. Eliot)
- Persuasion (Jane Austen)
"""

import requests
from pathlib import Path


def download_text(url: str, output_path: str, title: str):
    """Download text from URL and save to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading {title}...")
    response = requests.get(url)
    response.encoding = 'utf-8'

    # Extract text content (Project Gutenberg has headers/footers)
    text = response.text

    # Remove Project Gutenberg header/footer boilerplate
    start_marker = "***START"
    end_marker = "***END"

    if start_marker in text and end_marker in text:
        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)
        text = text[start_idx + len(start_marker):end_idx]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    lines = text.count("\n")
    print(f"   ✅ Saved to {output_path} ({lines} lines)")


if __name__ == "__main__":
    # Project Gutenberg URLs (plain text UTF-8)
    texts = [
        ("https://www.gutenberg.org/cache/epub/21373/pg21373.txt", "data/velveteen_rabbit.txt", "The Velveteen Rabbit"),
        ("https://www.gutenberg.org/cache/epub/84/pg84.txt", "data/frankenstein.txt", "Frankenstein"),
        ("https://www.gutenberg.org/cache/epub/1321/pg1321.txt", "data/waste_land.txt", "The Waste Land"),
        ("https://www.gutenberg.org/cache/epub/105/pg105.txt", "data/persuasion.txt", "Persuasion"),
    ]

    print("=" * 70)
    print("DOWNLOADING TEXTS FROM PROJECT GUTENBERG")
    print("=" * 70)
    print()

    for url, output_path, title in texts:
        try:
            download_text(url, output_path, title)
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print()
    print("=" * 70)
    print("✅ Downloads complete")
    print("=" * 70)
    print()
    print("Next: Train personas with:")
    print("  python -m toys.train_persona --text-path data/velveteen_rabbit.txt --persona-name velveteen")
    print("  python -m toys.train_persona --text-path data/frankenstein.txt --persona-name frankenstein")
    print("  python -m toys.train_persona --text-path data/waste_land.txt --persona-name waste_land")
    print("  python -m toys.train_persona --text-path data/persuasion.txt --persona-name persuasion")
