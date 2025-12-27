#!/usr/bin/env python3
"""
adventure_narrative_corpus.py: Build training corpus for puzzle narrative CharLSTM

Combines:
- Text adventure game vocabulary (Zork, Colossal Cave, etc.)
- D&D-style fantasy phrases
- Puzzle-specific narrative templates
- Early UNIX/DOS game linguistic patterns

Target: ~2MB CharLSTM model for on-device narrative generation

SPDX-License-Identifier: MIT
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# =============================================================================
# DSL VOCABULARY: Core phrases for puzzle adventure narratives
# =============================================================================

# Root action verbs (text adventure heritage)
ACTION_VERBS = [
    "examine", "inspect", "study", "observe", "analyze",
    "solve", "unlock", "discover", "reveal", "uncover",
    "navigate", "traverse", "explore", "venture", "journey",
    "decipher", "decode", "interpret", "translate", "comprehend",
    "master", "conquer", "overcome", "triumph", "prevail",
]

# Fantasy/D&D adjectives
FANTASY_ADJECTIVES = [
    "ancient", "mystical", "arcane", "forgotten", "legendary",
    "enchanted", "cursed", "blessed", "sacred", "profane",
    "crystalline", "obsidian", "ethereal", "spectral", "shadowy",
    "radiant", "luminous", "twilight", "starlit", "moonlit",
    "runic", "sigiled", "warded", "sealed", "bound",
]

# Puzzle-themed nouns
PUZZLE_NOUNS = [
    "grid", "matrix", "cipher", "puzzle", "enigma",
    "riddle", "labyrinth", "maze", "pattern", "sequence",
    "cage", "cell", "chamber", "vault", "sanctum",
    "theorem", "equation", "formula", "calculation", "sum",
    "glyph", "symbol", "rune", "sigil", "mark",
]

# Location descriptors (text adventure style)
LOCATIONS = [
    "chamber", "hall", "corridor", "passage", "alcove",
    "tower", "dungeon", "crypt", "library", "observatory",
    "garden", "courtyard", "throne room", "sanctum", "vault",
    "cavern", "grotto", "temple", "shrine", "altar",
]

# Character archetypes
CHARACTERS = [
    "sage", "wizard", "scholar", "mystic", "oracle",
    "knight", "guardian", "sentinel", "warden", "keeper",
    "apprentice", "adept", "master", "grandmaster", "archmage",
    "traveler", "seeker", "quester", "pilgrim", "wanderer",
]

# Temporal phrases (D&D campaign style)
TEMPORAL_PHRASES = [
    "In ages past", "Long ago", "In the time before",
    "When the stars aligned", "As the moon waxed full",
    "In the twilight hours", "At the break of dawn",
    "During the age of", "In the era of", "Before the fall of",
]

# Victory/completion phrases
VICTORY_PHRASES = [
    "The pattern resolves", "The seal breaks", "Light floods the chamber",
    "The runes pulse with power", "Ancient mechanisms stir",
    "The cipher yields its secrets", "Victory is assured",
    "The grid harmonizes", "Balance is restored", "The puzzle yields",
]

# =============================================================================
# NARRATIVE TEMPLATES: Structured patterns for generation
# =============================================================================

@dataclass
class NarrativeTemplate:
    """Template for generating narrative text."""
    pattern: str
    tags: List[str] = field(default_factory=list)

INTRO_TEMPLATES = [
    NarrativeTemplate(
        "{temporal} in the {adj} {location}, a {noun} of {difficulty} awaited discovery.",
        ["intro", "setting"]
    ),
    NarrativeTemplate(
        "The {character} spoke: '{action} the {adj} {noun}, and prove your worth.'",
        ["intro", "dialogue"]
    ),
    NarrativeTemplate(
        "Before you lies a {adj} {noun}. {size} cells form its {adj2} pattern.",
        ["intro", "puzzle_desc"]
    ),
    NarrativeTemplate(
        "Within the {adj} {location}, the {noun} pulses with {adj2} energy.",
        ["intro", "mystical"]
    ),
    NarrativeTemplate(
        "The ancient {character} left this {noun} as a test. Only the worthy may {action}.",
        ["intro", "challenge"]
    ),
    NarrativeTemplate(
        "Dust motes dance in the {adj} light. A {noun} covers the {location} floor.",
        ["intro", "atmospheric"]
    ),
    NarrativeTemplate(
        "You enter the {location}. Numbers shimmer within the {adj} {noun}.",
        ["intro", "text_adventure"]
    ),
    NarrativeTemplate(
        "A voice echoes: 'The {noun} of {difficulty} has claimed many. Will you {action}?'",
        ["intro", "ominous"]
    ),
]

OUTRO_TEMPLATES = [
    NarrativeTemplate(
        "{victory}! The {adj} {noun} surrenders to your logic.",
        ["outro", "victory"]
    ),
    NarrativeTemplate(
        "In {time} heartbeats, you have {action}ed the {adj} {noun}.",
        ["outro", "timed"]
    ),
    NarrativeTemplate(
        "The {character} nods approvingly. 'You have proven yourself.'",
        ["outro", "dialogue"]
    ),
    NarrativeTemplate(
        "{victory}. Another step on the path to mastery.",
        ["outro", "progress"]
    ),
    NarrativeTemplate(
        "The {location} trembles as the final number falls into place.",
        ["outro", "dramatic"]
    ),
    NarrativeTemplate(
        "Silence. Then a click. The {noun} is no more - only solution remains.",
        ["outro", "text_adventure"]
    ),
    NarrativeTemplate(
        "You have earned {reward}. The journey continues.",
        ["outro", "reward"]
    ),
]

# =============================================================================
# TEXT ADVENTURE VOCABULARY: Classic game phrases
# =============================================================================

# Classic text adventure commands repurposed as narrative elements
TEXT_ADVENTURE_VOCAB = {
    "directions": ["north", "south", "east", "west", "up", "down"],
    "examination": [
        "You see", "There is", "Upon closer inspection",
        "Examining reveals", "You notice", "It appears to be",
    ],
    "inventory": [
        "You carry", "In your possession", "Your mind holds",
        "You have learned", "Knowledge gained", "Wisdom acquired",
    ],
    "responses": [
        "I don't understand", "That's not possible here",
        "Nothing happens", "You can't do that", "Try again",
    ],
    "success": [
        "Done.", "OK.", "Taken.", "Dropped.", "Opened.",
        "Solved.", "Correct.", "Verified.", "Complete.", "Victory.",
    ],
    "atmosphere": [
        "It is dark here", "A cold wind blows", "Torches flicker",
        "Shadows dance", "Silence reigns", "Time stands still",
    ],
}

# Zork-inspired flavor text
ZORK_PHRASES = [
    "You are in a maze of twisty little passages, all alike.",
    "It is pitch dark. You are likely to be eaten by a grue.",
    "The door is locked. A puzzle guards its secrets.",
    "Your lamp is getting dim.",
    "You have moved up in rank.",
    "Your score is {score} out of a possible {max}.",
]

# =============================================================================
# D&D VOCABULARY: Fantasy RPG language
# =============================================================================

DND_VOCAB = {
    "ability_scores": ["Strength", "Dexterity", "Constitution", "Intelligence", "Wisdom", "Charisma"],
    "dice_references": [
        "Roll for insight", "Make a check", "Test your skill",
        "Fortune favors the bold", "The dice have spoken",
    ],
    "alignments": [
        "lawful and ordered", "chaotic yet brilliant", "neutral balance",
        "good prevails", "darkness retreats",
    ],
    "classes": [
        "like a wizard studying ancient tomes",
        "with a rogue's cunning",
        "displaying bardic wit",
        "channeling a cleric's wisdom",
    ],
    "campaign_phrases": [
        "Roll for initiative", "Natural twenty", "Critical success",
        "The DM smiles", "Quest complete", "Level up",
    ],
}

# =============================================================================
# CORPUS GENERATION
# =============================================================================

def generate_narrative(
    template: NarrativeTemplate,
    grid_size: int = 6,
    difficulty: str = "moderate",
    time_seconds: int = 120,
    is_victory: bool = True
) -> str:
    """Generate a narrative string from a template."""

    # Build substitution dictionary
    subs = {
        "temporal": random.choice(TEMPORAL_PHRASES),
        "adj": random.choice(FANTASY_ADJECTIVES),
        "adj2": random.choice(FANTASY_ADJECTIVES),
        "location": random.choice(LOCATIONS),
        "noun": random.choice(PUZZLE_NOUNS),
        "character": random.choice(CHARACTERS),
        "action": random.choice(ACTION_VERBS),
        "victory": random.choice(VICTORY_PHRASES),
        "difficulty": difficulty,
        "size": str(grid_size),
        "time": str(time_seconds),
        "reward": f"the Mark of the {random.choice(FANTASY_ADJECTIVES).title()} {random.choice(CHARACTERS).title()}",
        "score": str(random.randint(50, 100)),
        "max": "100",
    }

    # Perform substitution
    result = template.pattern
    for key, value in subs.items():
        result = result.replace("{" + key + "}", value)

    return result


def generate_corpus(
    num_samples: int = 10000,
    output_path: Optional[Path] = None
) -> List[str]:
    """Generate a training corpus of narrative texts."""

    corpus = []

    # Generate from templates
    all_templates = INTRO_TEMPLATES + OUTRO_TEMPLATES
    for _ in range(num_samples // 2):
        template = random.choice(all_templates)
        grid_size = random.choice([4, 5, 6, 7, 8, 9])
        difficulty = random.choice(["trivial", "easy", "moderate", "challenging", "legendary"])
        time_seconds = random.randint(30, 600)

        narrative = generate_narrative(template, grid_size, difficulty, time_seconds)
        corpus.append(narrative)

    # Add text adventure vocabulary
    for category, phrases in TEXT_ADVENTURE_VOCAB.items():
        for phrase in phrases:
            # Create variations
            for _ in range(num_samples // (len(TEXT_ADVENTURE_VOCAB) * 20)):
                if "{" not in phrase:
                    corpus.append(phrase)

    # Add Zork phrases
    corpus.extend(ZORK_PHRASES * (num_samples // 100))

    # Add D&D vocabulary
    for category, phrases in DND_VOCAB.items():
        corpus.extend(phrases * (num_samples // (len(DND_VOCAB) * 10)))

    # Add combined phrases (cross-pollination)
    for _ in range(num_samples // 4):
        parts = [
            random.choice(TEMPORAL_PHRASES),
            f"in the {random.choice(FANTASY_ADJECTIVES)} {random.choice(LOCATIONS)},",
            f"a {random.choice(CHARACTERS)}",
            f"must {random.choice(ACTION_VERBS)}",
            f"the {random.choice(FANTASY_ADJECTIVES)} {random.choice(PUZZLE_NOUNS)}.",
        ]
        corpus.append(" ".join(parts))

    # Shuffle
    random.shuffle(corpus)

    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in corpus:
                f.write(line + "\n")
        print(f"Saved {len(corpus)} samples to {output_path}")

    return corpus


def build_vocabulary(corpus: List[str]) -> Dict[str, int]:
    """Build character-level vocabulary from corpus."""

    chars = set()
    for text in corpus:
        chars.update(text)

    # Sort for reproducibility
    chars = sorted(chars)

    # Build vocab with special tokens
    vocab = {
        "<pad>": 0,
        "<sos>": 1,  # Start of sequence
        "<eos>": 2,  # End of sequence
        "<unk>": 3,  # Unknown character
    }

    for i, char in enumerate(chars, start=len(vocab)):
        vocab[char] = i

    return vocab


def export_vocabulary(vocab: Dict[str, int], output_path: Path) -> None:
    """Export vocabulary to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f"Saved vocabulary ({len(vocab)} tokens) to {output_path}")


# =============================================================================
# DSL PHRASE DATABASE: Compact storage for template-based generation
# =============================================================================

def export_phrase_database(output_path: Path) -> None:
    """Export phrase database for on-device template generation.

    This is an alternative to neural generation - uses structured
    template filling with random selection. Much smaller than a model.
    """

    database = {
        "version": 1,
        "action_verbs": ACTION_VERBS,
        "adjectives": FANTASY_ADJECTIVES,
        "nouns": PUZZLE_NOUNS,
        "locations": LOCATIONS,
        "characters": CHARACTERS,
        "temporal": TEMPORAL_PHRASES,
        "victory": VICTORY_PHRASES,
        "text_adventure": TEXT_ADVENTURE_VOCAB,
        "dnd": DND_VOCAB,
        "intro_patterns": [t.pattern for t in INTRO_TEMPLATES],
        "outro_patterns": [t.pattern for t in OUTRO_TEMPLATES],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(database, f, indent=2, ensure_ascii=False)

    size_kb = output_path.stat().st_size / 1024
    print(f"Saved phrase database ({size_kb:.1f} KB) to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate corpus and vocabulary for CharLSTM training."""

    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"

    print("=" * 60)
    print("Adventure Narrative Corpus Generator")
    print("=" * 60)

    # Generate corpus
    print("\n[1/4] Generating training corpus...")
    corpus = generate_corpus(
        num_samples=20000,
        output_path=data_dir / "narrative_corpus.txt"
    )

    # Build vocabulary
    print("\n[2/4] Building character vocabulary...")
    vocab = build_vocabulary(corpus)
    export_vocabulary(vocab, data_dir / "narrative_vocab.json")

    # Export phrase database (fallback/hybrid approach)
    print("\n[3/4] Exporting phrase database...")
    export_phrase_database(data_dir / "phrase_database.json")

    # Statistics
    print("\n[4/4] Corpus statistics:")
    total_chars = sum(len(text) for text in corpus)
    unique_chars = len(vocab) - 4  # Exclude special tokens
    avg_length = total_chars / len(corpus)

    print(f"  Total samples: {len(corpus):,}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Unique characters: {unique_chars}")
    print(f"  Average sample length: {avg_length:.1f} chars")
    print(f"  Vocabulary size: {len(vocab)} tokens")

    print("\n" + "=" * 60)
    print("Done! Ready for CharLSTM training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
