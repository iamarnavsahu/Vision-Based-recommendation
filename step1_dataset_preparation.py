"""Step 1: Dataset Preparation
------------------------------
Maps Places365 categories to 6 mood labels and creates organised train/val
symlink directories ready for PyTorch ImageFolder.

Usage
-----
    python step1_dataset_preparation.py

    # Custom paths / splits
    python step1_dataset_preparation.py \\
        --places365_dir data_256 \\
        --categories_file categories_places365.txt \\
        --output_dir mood_dataset \\
        --val_split 0.2 \\
        --images_per_class 5000

Output
------
    mood_dataset/
        train/  calm/ cheerful/ energetic/ dark_tense/ dramatic/ melancholic/
        val/    calm/ cheerful/ energetic/ dark_tense/ dramatic/ melancholic/
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Mood taxonomy
# ---------------------------------------------------------------------------

MOOD_LABELS: list[str] = [
    "calm",
    "cheerful",
    "energetic",
    "dark_tense",
    "dramatic",
    "melancholic",
]

# Places365 category name substrings → mood.
# Matching is done by checking if any keyword is a substring of the category
# name, or vice versa.  Ordered from most specific to least specific so that
# the first match wins.
MOOD_KEYWORDS: dict[str, list[str]] = {
    "calm": [
        "lake", "pond", "river", "creek", "stream", "bayou", "delta",
        "lagoon", "tide_flat", "coral_reef", "hot_spring",
        "forest", "woodland", "rainforest", "bamboo", "grove",
        "meadow", "field", "pasture", "prairie", "orchard", "vineyard",
        "garden", "zen", "park", "botanical_garden",
        "valley", "countryside", "rural", "farmland",
        "swamp", "marsh", "wetland",
        "campsite", "picnic",
    ],
    "cheerful": [
        "beach", "amusement_park", "playground", "market", "bazaar",
        "boardwalk", "pier", "resort", "water_park", "aquarium",
        "food_court", "restaurant", "cafe", "bakery", "ice_cream",
        "shopping_mall", "toy_store", "candy_store",
        "fair", "carnival",
        "swimming_pool", "waterpark",
    ],
    "energetic": [
        "stadium", "gym", "gymnasium", "arena",
        "race_track", "racetrack", "speedway",
        "athletic_field", "basketball_court", "tennis_court",
        "soccer_field", "baseball_field", "football_field",
        "volleyball_court", "boxing_ring", "skatepark", "climbing_wall",
        "bowling_alley", "golf_course", "ski_slope", "ski_resort",
        "crossfit", "weight_room",
    ],
    "dark_tense": [
        "alley", "back_alley",
        "tunnel", "underpass",
        "jail", "prison", "correctional",
        "basement", "cellar", "dungeon",
        "cave", "mine", "mineshaft",
        "industrial", "factory", "boiler_room", "power_plant",
        "nuclear", "chemical_plant",
        "wasteland", "junkyard", "salvage_yard", "landfill",
        "demolition", "slum", "ghetto",
        "graffiti", "sewer", "parking_garage",
        "abandoned",
    ],
    "dramatic": [
        "cliff", "sea_cliff", "rock_arch",
        "volcano", "crater",
        "mountain", "mountain_peak", "summit",
        "glacier", "iceberg",
        "canyon", "gorge", "ravine",
        "desert", "badlands", "mesa", "butte", "dune",
        "castle", "fortress", "citadel",
        "cathedral", "basilica",
        "temple", "ruins", "ruin",
        "skyscraper", "tower", "dam", "bridge", "monument", "palace",
        "ocean", "sea", "fjord",
    ],
    "melancholic": [
        "cemetery", "graveyard", "memorial",
        "hospital", "operating_room", "nursing_home",
        "empty_room", "corridor", "hallway",
        "attic", "storage_room",
        "laundromat", "waiting_room",
        "bus_station", "train_station", "airport_terminal",
        "foggy", "fog", "rain", "mist", "overcast",
        "fire_escape", "apartment_building",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_categories(categories_file: Path) -> list[str]:
    """Load Places365 category names from the categories file.

    Handles two formats:
    - ``/a/abbey 0``  (Places365 standard format)
    - ``abbey``       (plain name)

    Args:
        categories_file: Path to ``categories_places365.txt``.

    Returns:
        List of lowercase category name strings.
    """
    categories: list[str] = []
    with open(categories_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            token = line.split()[0]
            if token.startswith("/"):
                name = token.rstrip("/").split("/")[-1]
            else:
                name = token
            categories.append(name.lower())
    return categories


def assign_mood(category: str) -> str | None:
    """Assign a mood label to a Places365 category via keyword matching.

    The first matching mood in ``MOOD_KEYWORDS`` wins, so ordering matters.

    Args:
        category: Lowercase Places365 category name (underscores, no spaces).

    Returns:
        Mood label string, or ``None`` if no keyword matches.
    """
    for mood, keywords in MOOD_KEYWORDS.items():
        for kw in keywords:
            if kw in category or category in kw:
                return mood
    return None


def _find_category_dir(places365_dir: Path, cat: str) -> Path | None:
    """Locate the image directory for a Places365 category.

    Places365 datasets use different structures:
    - ``data_256/<category>/``
    - ``data_256/<first_letter>/<category>/``

    Args:
        places365_dir: Root Places365 directory.
        cat: Category name.

    Returns:
        Path to the category directory, or ``None`` if not found.
    """
    for candidate in [
        places365_dir / cat,
        places365_dir / cat[0] / cat,
    ]:
        if candidate.is_dir():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Core dataset builder
# ---------------------------------------------------------------------------


def build_mood_dataset(
    places365_dir: Path,
    output_dir: Path,
    categories: list[str],
    val_split: float = 0.2,
    seed: int = 42,
    images_per_class: int = 5000,
) -> dict[str, dict[str, int]]:
    """Build the mood dataset from Places365 source images.

    Creates ``output_dir/train/<mood>/`` and ``output_dir/val/<mood>/``
    directories populated with symlinks (or copies when symlinks fail) to
    the original Places365 images.

    Args:
        places365_dir: Root directory of the Places365 dataset.
        output_dir: Where to write ``train/`` and ``val/`` splits.
        categories: List of Places365 category names.
        val_split: Fraction of images per category reserved for validation.
        seed: Random seed for reproducibility.
        images_per_class: Maximum images to include per mood class in total.

    Returns:
        Nested dict ``counts[split][mood] = image_count``.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Map each category to its mood
    cat_to_mood: dict[str, str] = {}
    for cat in categories:
        mood = assign_mood(cat)
        if mood:
            cat_to_mood[cat] = mood

    print(f"Mapped {len(cat_to_mood)} / {len(categories)} categories to moods.\n")
    for mood in MOOD_LABELS:
        matched = [c for c, m in cat_to_mood.items() if m == mood]
        print(f"  {mood:15s} → {len(matched):3d} categories  (e.g. {matched[:4]})")

    # Create output directory tree
    for split in ("train", "val"):
        for mood in MOOD_LABELS:
            (output_dir / split / mood).mkdir(parents=True, exist_ok=True)

    counts: dict[str, dict[str, int]] = {
        "train": defaultdict(int),
        "val": defaultdict(int),
    }

    # Per-mood image caps to keep class balance
    mood_budgets: dict[str, int] = {m: images_per_class for m in MOOD_LABELS}

    for cat, mood in tqdm(cat_to_mood.items(), desc="Linking images"):
        if mood_budgets[mood] <= 0:
            continue

        cat_dir = _find_category_dir(places365_dir, cat)
        if cat_dir is None:
            continue

        imgs = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png"))
        random.shuffle(imgs)

        # Respect remaining budget for this mood
        imgs = imgs[: mood_budgets[mood]]
        mood_budgets[mood] -= len(imgs)

        n_val = max(1, int(len(imgs) * val_split))
        splits_imgs = {
            "val": imgs[:n_val],
            "train": imgs[n_val:],
        }

        for split, files in splits_imgs.items():
            for img_path in files:
                dst = output_dir / split / mood / f"{cat}__{img_path.name}"
                if dst.exists():
                    continue
                try:
                    os.symlink(img_path.resolve(), dst)
                except OSError:
                    shutil.copy2(img_path, dst)
                counts[split][mood] += 1

    return counts


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(counts: dict[str, dict[str, int]]) -> None:
    """Print a formatted table of image counts per split and mood class.

    Args:
        counts: Nested dict returned by :func:`build_mood_dataset`.
    """
    print("\n" + "=" * 50)
    print("Dataset Summary")
    print("=" * 50)
    print(f"{'Mood':<15} {'Train':>8} {'Val':>8} {'Total':>8}")
    print("-" * 50)
    for mood in MOOD_LABELS:
        tr = counts["train"].get(mood, 0)
        va = counts["val"].get(mood, 0)
        print(f"{mood:<15} {tr:>8,} {va:>8,} {tr + va:>8,}")
    print("-" * 50)
    total_tr = sum(counts["train"].values())
    total_va = sum(counts["val"].values())
    print(f"{'TOTAL':<15} {total_tr:>8,} {total_va:>8,} {total_tr + total_va:>8,}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare mood dataset from Places365.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--places365_dir",
        type=Path,
        default=Path("data_256"),
        help="Root directory of the Places365 256×256 dataset.",
    )
    parser.add_argument(
        "--categories_file",
        type=Path,
        default=Path("categories_places365.txt"),
        help="Path to categories_places365.txt.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("mood_dataset"),
        help="Output directory for train/val splits.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of images per category used for validation.",
    )
    parser.add_argument(
        "--images_per_class",
        type=int,
        default=5000,
        help="Maximum total images per mood class.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.categories_file.exists():
        raise FileNotFoundError(
            f"{args.categories_file} not found.\n"
            "Download it with:\n"
            "  wget https://raw.githubusercontent.com/CSAILVision/places365"
            "/master/categories_places365.txt"
        )

    if not args.places365_dir.exists():
        raise FileNotFoundError(
            f"{args.places365_dir} not found.\n"
            "Download Places365 (256×256) from:\n"
            "  http://places2.csail.mit.edu/download.html"
        )

    print(f"Places365 dir  : {args.places365_dir.resolve()}")
    print(f"Categories file: {args.categories_file.resolve()}")
    print(f"Output dir     : {args.output_dir.resolve()}")
    print(f"Val split      : {args.val_split}")
    print(f"Max per class  : {args.images_per_class:,}\n")

    categories = load_categories(args.categories_file)
    print(f"Loaded {len(categories)} Places365 categories.\n")

    counts = build_mood_dataset(
        places365_dir=args.places365_dir,
        output_dir=args.output_dir,
        categories=categories,
        val_split=args.val_split,
        seed=args.seed,
        images_per_class=args.images_per_class,
    )

    print_summary(counts)
    print(f"\nDone.  Dataset written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
