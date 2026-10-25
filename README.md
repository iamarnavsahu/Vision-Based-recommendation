# ECE 176 Final Project( RAGA AI): Vision-Based Music Recommendation

**Arnav Sahu — UC San Diego**

## Project Overview

Upload a video → AI predicts the mood → Get matching music recommendations.
This is the vision engine that will power RAGA AI's cinematic music generation pipeline.

---

## Quick Start

### Step 0: Environment Setup

```bash
# Create conda environment (recommended)
conda create -n ece176 python=3.10
conda activate ece176

# Install PyTorch (check https://pytorch.org for your CUDA version)
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

### Step 1: Dataset Preparation

```bash
# Download the Places365 category list
wget https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt

# Download Places365 Standard (256x256) — ~21GB
# Go to http://places2.csail.mit.edu/download.html
# Download "Small images (256*256) with easy directory structure"

# Run the dataset preparation script
python step1_dataset_preparation.py
```

**Output:** `mood_dataset/` with `train/` and `val/` splits across 6 mood classes (~30K images)

### Step 2: Train Baseline (Experiment 1)

```bash
python step2_model_training.py
```

**Output:**
- `checkpoints/best_model.pth` — trained ResNet-50 mood classifier
- `results/exp1_training_curves.png` — loss/accuracy plots
- `results/exp1_confusion_matrix.png` — confusion matrix

### Step 3: Add Color & Motion Features (Experiment 2)

```bash
python step3_color_motion_features.py
```

### Step 4: Test Smaller Backbone (Experiment 3)

```bash
python step2_model_training.py --backbone resnet18
```

### Step 5: Build Music Library & End-to-End Pipeline

```bash
python step5_pipeline.py --video path/to/your/video.mp4
```

---

## Project Structure

```
ece176-project/
├── step1_dataset_preparation.py   # Places365 → mood dataset
├── step2_model_training.py        # Experiment 1: ResNet-50 baseline
├── step3_color_motion_features.py # Experiment 2: + color/motion features
├── step5_pipeline.py              # Full pipeline: video → mood → music
├── requirements.txt
├── categories_places365.txt       # Places365 category names
│
├── mood_dataset/                  # Created by step 1
│   ├── train/
│   │   ├── calm/
│   │   ├── cheerful/
│   │   ├── dark_tense/
│   │   ├── dramatic/
│   │   ├── energetic/
│   │   └── melancholic/
│   └── val/
│
├── checkpoints/                   # Saved model weights
├── results/                       # Plots, metrics, confusion matrices
└── music_library/                 # Royalty-free tracks (built in step 5)
    └── library.csv                # track_name, mood_tag, tempo, file_path
```

---

## Timeline

| Date   | Milestone                                      |
|--------|------------------------------------------------|
| Feb 25 | Dataset ready, baseline training running       |
| Mar 4  | All 3 experiments done with results            |
| Mar 11 | Music library + end-to-end demo working        |
| Mar 15 | Report + demo video submitted                  |

---

## Mood Categories

| Mood        | Description                        | Example Scenes                          |
|-------------|------------------------------------|-----------------------------------------|
| Calm        | Peaceful, serene, tranquil         | Lakes, forests, zen gardens             |
| Cheerful    | Happy, warm, bright                | Beaches, playgrounds, markets           |
| Energetic   | High-energy, fast, intense         | Stadiums, concerts, race tracks         |
| Dark/Tense  | Ominous, suspenseful, gritty       | Alleys, caves, industrial areas         |
| Dramatic    | Epic, grand, awe-inspiring         | Cliffs, volcanoes, castles              |
| Melancholic | Sad, lonely, reflective            | Cemeteries, empty corridors, rain       |

---

