# CLAUDE.md — RAGA AI: Vision-Driven Music Recommendation

> **ECE 176 Final Project** | Deadline: March 15, 2026
> **Author:** Arnav Sahu (`iamarnavsahu`)
> **Working branch:** `claude/claude-md-mlud1anuqgnuvhdz-7fREy`

---

## Project Overview

RAGA AI is a vision-driven music recommendation system for content creators (YouTubers, filmmakers, etc.). It analyzes visual content — video frames or images — infers the emotional mood, and recommends royalty-free music tracks that match that mood.

### The Problem Being Solved

Creators need music for their videos but face two pain points:
1. **Licensing complexity** — recommending existing commercial music means negotiating rights for every track.
2. **Subjective matching** — finding music that "fits" a video is tedious and manual.

### Solution (ECE 176 Scope)

```
Input Frames / Images
        ↓
  Visual Analysis
  (CNN / CLIP / ViT)
        ↓
   Mood Inference
  (e.g. calm, epic,
   melancholic, tense)
        ↓
 Royalty-Free Library
   (Pixabay, FMA)
        ↓
  Ranked Music Recs
```

Rights are solved by restricting the recommendation pool to **explicitly royalty-free sources** (Pixabay Audio, Free Music Archive). No third-party licensing issues.

### Future Direction (RAGA Pivot — Post-Course)

Replace the lookup step with conditional **AI music generation** (e.g., Meta MusicGen, MusicLM):

```
Frames → Mood → MusicGen(mood_condition) → Original Audio Clip
```

Generated music is original content — sidesteps licensing entirely and enables cinematic, mood-matched scores on demand.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Vision backbone | CLIP (`openai/clip-vit-base-patch32`) or ResNet-50 |
| Mood classifier | Fine-tuned linear head on top of vision embeddings |
| Music metadata | JSON / CSV index of royalty-free tracks |
| Similarity search | Cosine similarity or FAISS |
| API / Demo | FastAPI + Gradio (or Streamlit) |
| Deep learning | PyTorch |
| Experiment tracking | (optional) Weights & Biases or TensorBoard |
| Package management | `pip` + `requirements.txt` |
| Linting / formatting | `black`, `isort`, `flake8` |

---

## Planned Repository Structure

```
Final_project/
├── CLAUDE.md                  # This file — AI assistant context
├── README.md                  # Human-facing project description
├── requirements.txt           # Python dependencies
├── .gitignore
│
├── data/
│   ├── frames/                # Sample extracted video frames (JPG/PNG)
│   ├── music_index.json       # Metadata for royalty-free tracks
│   │                          #   {id, title, source_url, mood_tags, bpm, duration}
│   └── splits/                # train/val/test splits for mood classifier
│
├── src/
│   ├── __init__.py
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── extractor.py       # Frame extraction from video (ffmpeg wrapper)
│   │   └── encoder.py         # CLIP / ResNet embedding generation
│   │
│   ├── mood/
│   │   ├── __init__.py
│   │   ├── classifier.py      # Mood classification head (linear / MLP)
│   │   ├── labels.py          # Mood taxonomy (calm, epic, melancholic, tense, happy, dark)
│   │   └── train.py           # Training loop for mood classifier
│   │
│   ├── recommendation/
│   │   ├── __init__.py
│   │   ├── index.py           # Load + index royalty-free music library
│   │   └── recommender.py     # Mood → ranked track list
│   │
│   └── api/
│       ├── __init__.py
│       ├── app.py             # FastAPI application
│       └── demo.py            # Gradio / Streamlit demo UI
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_clip_embeddings.ipynb
│   ├── 03_mood_classifier.ipynb
│   └── 04_end_to_end_demo.ipynb
│
├── tests/
│   ├── test_encoder.py
│   ├── test_classifier.py
│   └── test_recommender.py
│
└── scripts/
    ├── download_music_index.py  # Fetch metadata from Pixabay / FMA
    ├── extract_frames.py        # Batch frame extraction
    └── evaluate.py              # Compute recommendation metrics
```

---

## Core Modules — What Each Does

### `src/vision/encoder.py`
- Loads a pre-trained CLIP or ResNet model.
- `encode_image(image_path) -> np.ndarray` — returns a normalized embedding vector.
- `encode_frames(frame_dir) -> np.ndarray` — encodes multiple frames and returns mean-pooled embedding.

### `src/mood/classifier.py`
- Takes an image embedding and predicts one of N mood classes.
- `predict_mood(embedding) -> str` — returns mood label.
- `predict_mood_probs(embedding) -> dict[str, float]` — returns class probabilities (useful for soft matching).

### `src/mood/labels.py`
Defines the **mood taxonomy** used throughout the project:
```python
MOOD_LABELS = ["calm", "epic", "melancholic", "tense", "happy", "dark", "romantic", "energetic"]
```
All data labeling, model outputs, and music tags must use exactly these strings.

### `src/recommendation/index.py`
- Loads `data/music_index.json`.
- Indexes tracks by mood tags.
- `get_tracks_by_mood(mood: str, top_k: int) -> list[dict]`

### `src/recommendation/recommender.py`
- Orchestrates the full pipeline: frames → embedding → mood → tracks.
- `recommend(image_path_or_dir, top_k=5) -> list[dict]`

---

## Development Workflow

### Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the Demo

```bash
# Gradio/Streamlit UI
python src/api/demo.py

# FastAPI server
uvicorn src.api.app:app --reload --port 8000
```

### Training the Mood Classifier

```bash
python src/mood/train.py \
  --data_dir data/splits \
  --epochs 20 \
  --lr 1e-3 \
  --output_dir checkpoints/
```

### Running Tests

```bash
pytest tests/ -v
```

### Linting / Formatting

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/ --max-line-length=100
```

---

## Data Sources

| Source | Type | License | Notes |
|---|---|---|---|
| Pixabay Audio | Music tracks | Royalty-free (Pixabay License) | Free for commercial use, no attribution required |
| Free Music Archive (FMA) | Music tracks | Various CC licenses | Use CC0 / CC BY tracks only |
| Custom frame dataset | Images | N/A (user-provided) | Frames extracted from public domain or owned videos |

### Music Index Schema (`data/music_index.json`)

```json
[
  {
    "id": "pixabay_12345",
    "title": "Epic Cinematic Rise",
    "source": "pixabay",
    "source_url": "https://pixabay.com/music/...",
    "mood_tags": ["epic", "tense"],
    "bpm": 128,
    "duration_sec": 120,
    "download_url": "..."
  }
]
```

---

## Mood Taxonomy & Labeling Conventions

The project uses **8 mood classes**. When labeling training data or tagging music tracks, use exactly these lowercase strings:

```
calm | epic | melancholic | tense | happy | dark | romantic | energetic
```

**Mapping guidelines:**
- `epic` — large orchestral swells, rising action, heroic themes
- `tense` — suspense, thriller, dissonant harmonies
- `calm` — ambient, lo-fi, peaceful nature scenes
- `melancholic` — minor keys, slow tempo, emotional/sad imagery
- `dark` — horror, heavy minor, ominous atmospheres
- `happy` — upbeat, major keys, bright scenes
- `romantic` — cinematic love themes, gentle strings
- `energetic` — fast tempo, upbeat, action scenes

---

## Key Conventions for AI Assistants

### Code Style
- **Python 3.10+** — use modern syntax (`match`, `|` union types, etc.).
- Line length: **100 characters** max.
- Formatter: `black`. Import sorter: `isort`.
- Type hints on all public function signatures.
- Docstrings on all public classes and functions (Google style).

### Imports
```python
# Standard library first
import os
from pathlib import Path

# Third-party
import numpy as np
import torch
from PIL import Image

# Internal
from src.mood.labels import MOOD_LABELS
```

### No Hardcoded Paths
Always use `Path(__file__).parent` or pass paths as arguments. Never hardcode absolute paths.

### Model Loading
Load models once at module level or in a singleton. Do not reload on every function call.

```python
# Good
_model = None
def get_model():
    global _model
    if _model is None:
        _model = load_clip()
    return _model
```

### Mood Labels Are a Constant
Import `MOOD_LABELS` from `src/mood/labels.py` everywhere. Do not redefine the list elsewhere.

### Tests
- Each public function in `src/` should have at least one test.
- Use `pytest` fixtures for model loading (mark with `@pytest.mark.slow` if loading a large model).
- Mock external HTTP calls (Pixabay API, FMA) in tests.

### Git Commits
- Format: `type(scope): short description`
- Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
- Examples:
  - `feat(mood): add linear classifier head on CLIP embeddings`
  - `fix(recommender): handle empty mood tag list`
  - `docs: update CLAUDE.md with music index schema`

### Branch Naming
- Feature branches: `feature/<short-description>`
- Bug fixes: `fix/<short-description>`
- AI assistant branches: `claude/<session-id>`

---

## Milestones (toward March 15 Deadline)

| Milestone | Target |
|---|---|
| Frame extractor + CLIP encoder working | Week 1 |
| Music index built (Pixabay + FMA metadata) | Week 1 |
| Mood classifier trained (even on small dataset) | Week 2 |
| End-to-end pipeline: image → recommendation | Week 2 |
| Gradio demo UI | Week 3 |
| Tests + cleanup + report | Week 3 |

---

## Future Work (Post-Course RAGA Pivot)

The ECE 176 project establishes the **vision → mood** pipeline. The next phase for RAGA replaces the recommendation lookup with AI music generation:

1. **Keep:** `src/vision/` and `src/mood/` — the mood signal is the generation condition.
2. **Add:** Integration with Meta [MusicGen](https://github.com/facebookresearch/audiocraft) or similar:
   ```python
   from audiocraft.models import MusicGen
   model = MusicGen.get_pretrained("facebook/musicgen-medium")
   model.set_generation_params(duration=30)
   wav = model.generate(["epic cinematic orchestral"])
   ```
3. **Result:** Original audio — no third-party rights involved. Define your own terms of use for the generated clips.

---

## Frequently Asked Questions for AI Assistants

**Q: Where is the entry point?**
`src/api/demo.py` for the UI demo; `src/api/app.py` for the FastAPI server; `src/recommendation/recommender.py` for the core pipeline.

**Q: What mood labels are valid?**
Exactly the 8 in `src/mood/labels.py`: `calm`, `epic`, `melancholic`, `tense`, `happy`, `dark`, `romantic`, `energetic`.

**Q: Which music sources are approved?**
Only Pixabay Audio and Free Music Archive (CC0/CC BY tracks). Do not add commercial or unlicensed tracks.

**Q: What vision model is preferred?**
CLIP (`openai/clip-vit-base-patch32`) is the default. ResNet-50 is the fallback if CLIP is too heavy.

**Q: How is the mood matched to music?**
Predicted mood label → filter `music_index.json` by `mood_tags` → rank by relevance (cosine similarity on mood embeddings or simple tag overlap).

**Q: Is this the generation version?**
No. This repo is the **recommendation** version (ECE 176 project). Generation (MusicGen) is future work for RAGA post-course.
