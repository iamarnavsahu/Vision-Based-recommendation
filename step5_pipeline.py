"""Step 5: End-to-End Pipeline — Video → Mood → Music Recommendations
---------------------------------------------------------------------
Runs the full inference pipeline:

    1. Extract frames from a video (or accept a directory of images).
    2. Classify the mood of each frame with the trained ResNet classifier.
    3. Aggregate per-frame predictions into a single mood label.
    4. Load the royalty-free music library (music_library/library.csv).
    5. Filter and rank tracks that match the predicted mood.
    6. Print the top-K recommendations (and optionally show a Gradio demo).

Usage
-----
    # Analyse a video
    python step5_pipeline.py --video path/to/video.mp4

    # Analyse an image directory
    python step5_pipeline.py --image_dir path/to/frames/

    # Specify checkpoint and top-K
    python step5_pipeline.py --video clip.mp4 \\
        --checkpoint checkpoints/best_model.pth \\
        --top_k 5

    # Launch Gradio demo UI
    python step5_pipeline.py --demo

Music Library CSV schema
------------------------
    track_name, mood_tag, tempo, duration_sec, source, source_url, file_path

    mood_tag must be one of the 6 mood classes produced by the classifier.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOOD_LABELS: list[str] = [
    "calm",
    "cheerful",
    "energetic",
    "dark_tense",
    "dramatic",
    "melancholic",
]
NUM_CLASSES = len(MOOD_LABELS)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MUSIC_LIBRARY_PATH = Path("music_library") / "library.csv"

# Default number of frames to sample from a video for efficiency
DEFAULT_FRAME_SAMPLE = 16

# ---------------------------------------------------------------------------
# Image transform (inference — no augmentation)
# ---------------------------------------------------------------------------

_infer_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model_cache: nn.Module | None = None
_class_names_cache: list[str] | None = None


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, list[str]]:
    """Load a trained mood classifier from a checkpoint.

    The checkpoint is expected to be a dict produced by step 2 or step 3 with
    keys ``model_state_dict``, ``backbone``, and ``class_names``.

    Args:
        checkpoint_path: Path to the ``.pth`` checkpoint file.
        device: Torch device to load the model onto.

    Returns:
        Tuple of (model in eval mode, list of class name strings).

    Raises:
        FileNotFoundError: If ``checkpoint_path`` does not exist.
    """
    global _model_cache, _class_names_cache

    if _model_cache is not None and _class_names_cache is not None:
        return _model_cache, _class_names_cache

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train a model first with:  python step2_model_training.py"
        )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    backbone_name = ckpt.get("backbone", "resnet50")
    class_names: list[str] = ckpt.get("class_names", MOOD_LABELS)

    backbone_fn = getattr(models, backbone_name)
    model = backbone_fn(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(class_names)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    _model_cache = model
    _class_names_cache = class_names
    return model, class_names


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def extract_frames_from_video(
    video_path: Path,
    n_frames: int = DEFAULT_FRAME_SAMPLE,
) -> list[np.ndarray]:
    """Uniformly sample N frames from a video file.

    Args:
        video_path: Path to the video file.
        n_frames: Number of frames to extract.

    Returns:
        List of RGB ``uint8`` numpy arrays, one per sampled frame.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If OpenCV cannot open the video.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"Could not read any frames from {video_path}")

    return frames


def load_frames_from_dir(image_dir: Path) -> list[np.ndarray]:
    """Load all images from a directory as RGB numpy arrays.

    Args:
        image_dir: Directory containing image files.

    Returns:
        List of RGB ``uint8`` numpy arrays.

    Raises:
        ValueError: If no supported images are found.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise ValueError(f"No images found in {image_dir}")

    frames: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


# ---------------------------------------------------------------------------
# Mood inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_mood_from_frames(
    frames: list[np.ndarray],
    model: nn.Module,
    class_names: list[str],
    device: torch.device,
    batch_size: int = 8,
) -> tuple[str, dict[str, float]]:
    """Predict the dominant mood from a list of frames.

    Runs the model on every frame, averages the softmax probabilities across
    all frames, and returns the top mood plus the full probability distribution.

    Args:
        frames: List of RGB numpy arrays.
        model: Loaded mood classifier in eval mode.
        class_names: Ordered list of class names matching model output indices.
        device: Torch device.
        batch_size: Frames per forward pass.

    Returns:
        Tuple of (predicted mood label, {class: probability} dict).
    """
    all_probs: list[torch.Tensor] = []

    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i : i + batch_size]
        tensors = [_infer_transform(Image.fromarray(f)) for f in batch_frames]
        batch = torch.stack(tensors).to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu())

    mean_probs = torch.cat(all_probs, dim=0).mean(dim=0)
    pred_idx = mean_probs.argmax().item()
    mood = class_names[pred_idx]
    prob_dict = {cls: float(mean_probs[i]) for i, cls in enumerate(class_names)}
    return mood, prob_dict


# ---------------------------------------------------------------------------
# Music library
# ---------------------------------------------------------------------------


def load_music_library(library_path: Path) -> pd.DataFrame:
    """Load the royalty-free music library CSV.

    Expected columns: track_name, mood_tag, tempo, duration_sec,
                      source, source_url, file_path.

    Args:
        library_path: Path to ``library.csv``.

    Returns:
        DataFrame of music tracks.

    Raises:
        FileNotFoundError: If the library file does not exist.
    """
    if not library_path.exists():
        raise FileNotFoundError(
            f"Music library not found: {library_path}\n"
            "Create it at music_library/library.csv with columns:\n"
            "  track_name, mood_tag, tempo, duration_sec, source, source_url, file_path"
        )
    return pd.read_csv(library_path)


def recommend_tracks(
    mood: str,
    library: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """Filter and rank music tracks for a given mood.

    Tracks are first filtered by exact ``mood_tag`` match, then sorted by
    ``tempo`` proximity to a mood-typical tempo target.

    Args:
        mood: Predicted mood label (must be one of ``MOOD_LABELS``).
        library: DataFrame returned by :func:`load_music_library`.
        top_k: Maximum number of recommendations to return.

    Returns:
        DataFrame of recommended tracks (at most ``top_k`` rows).
    """
    # Mood → typical tempo centre (BPM) for soft ranking
    mood_tempo: dict[str, int] = {
        "calm": 70,
        "cheerful": 110,
        "energetic": 140,
        "dark_tense": 90,
        "dramatic": 100,
        "melancholic": 75,
    }
    target_bpm = mood_tempo.get(mood, 100)

    matched = library[library["mood_tag"] == mood].copy()
    if matched.empty:
        return matched

    if "tempo" in matched.columns:
        matched["_bpm_dist"] = (matched["tempo"] - target_bpm).abs()
        matched = matched.sort_values("_bpm_dist").drop(columns=["_bpm_dist"])

    return matched.head(top_k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------


def print_mood_probabilities(probs: dict[str, float], predicted: str) -> None:
    """Print a bar-chart style probability breakdown."""
    print("\nMood probabilities:")
    for mood in sorted(probs, key=probs.get, reverse=True):
        bar = "█" * int(probs[mood] * 40)
        marker = " ◀ predicted" if mood == predicted else ""
        print(f"  {mood:15s} {probs[mood]:.3f}  {bar}{marker}")


def print_recommendations(tracks: pd.DataFrame) -> None:
    """Print the recommended tracks in a readable table."""
    if tracks.empty:
        print("\nNo matching tracks found in the library.")
        return

    print(f"\nTop {len(tracks)} music recommendations:\n")
    for i, row in tracks.iterrows():
        print(f"  {i + 1}. {row.get('track_name', 'Unknown')}")
        if "source" in row:
            print(f"     Source   : {row['source']}")
        if "tempo" in row:
            print(f"     Tempo    : {row['tempo']} BPM")
        if "duration_sec" in row:
            mins, secs = divmod(int(row["duration_sec"]), 60)
            print(f"     Duration : {mins}:{secs:02d}")
        if "source_url" in row:
            print(f"     URL      : {row['source_url']}")
        print()


# ---------------------------------------------------------------------------
# Gradio demo
# ---------------------------------------------------------------------------


def launch_gradio_demo(
    checkpoint_path: Path,
    library_path: Path,
    top_k: int,
) -> None:
    """Launch a Gradio web UI for the end-to-end pipeline.

    Args:
        checkpoint_path: Path to the trained model checkpoint.
        library_path: Path to the music library CSV.
        top_k: Number of recommendations to display.
    """
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run:  pip install gradio")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(checkpoint_path, device)

    try:
        library = load_music_library(library_path)
    except FileNotFoundError as exc:
        print(exc)
        library = pd.DataFrame(
            columns=["track_name", "mood_tag", "tempo", "duration_sec", "source", "source_url"]
        )

    def _predict_image(image: np.ndarray) -> tuple[str, str]:
        """Gradio callback: image → mood + track list string."""
        frames = [image]
        mood, probs = predict_mood_from_frames(frames, model, class_names, device)
        prob_lines = "\n".join(
            f"  {m}: {p:.3f}" for m, p in sorted(probs.items(), key=lambda x: -x[1])
        )
        recs = recommend_tracks(mood, library, top_k=top_k)
        if recs.empty:
            rec_text = "No tracks found for this mood."
        else:
            rec_lines = []
            for _, row in recs.iterrows():
                line = f"• {row.get('track_name', '?')}"
                if "tempo" in row:
                    line += f"  ({row['tempo']} BPM)"
                if "source_url" in row:
                    line += f"\n  {row['source_url']}"
                rec_lines.append(line)
            rec_text = "\n\n".join(rec_lines)

        return f"Predicted mood: **{mood}**\n\n{prob_lines}", rec_text

    demo = gr.Interface(
        fn=_predict_image,
        inputs=gr.Image(label="Upload a video frame or image", type="numpy"),
        outputs=[
            gr.Markdown(label="Mood prediction"),
            gr.Textbox(label="Music recommendations", lines=12),
        ],
        title="RAGA AI — Mood-Based Music Recommendation",
        description=(
            "Upload a video frame or image. The model will predict the mood "
            "and suggest royalty-free music tracks."
        ),
        flagging_mode="never",
    )
    demo.launch()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: video/images → mood → music recommendations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, help="Path to a video file.")
    source.add_argument("--image_dir", type=Path, help="Directory of image frames.")
    source.add_argument("--demo", action="store_true", help="Launch Gradio web UI.")

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints") / "best_model.pth",
        help="Trained model checkpoint.",
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=MUSIC_LIBRARY_PATH,
        help="Path to music_library/library.csv.",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of recommendations.")
    parser.add_argument(
        "--n_frames",
        type=int,
        default=DEFAULT_FRAME_SAMPLE,
        help="Frames to sample from the video.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Gradio demo mode
    # ------------------------------------------------------------------
    if args.demo:
        launch_gradio_demo(args.checkpoint, args.library, args.top_k)
        return

    if args.video is None and args.image_dir is None:
        parser.error("Provide --video, --image_dir, or --demo.")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, class_names = load_model(args.checkpoint, device)
    print(f"Model loaded from {args.checkpoint}  (classes: {class_names})\n")

    # ------------------------------------------------------------------
    # Extract frames
    # ------------------------------------------------------------------
    if args.video is not None:
        print(f"Extracting {args.n_frames} frames from {args.video} …")
        frames = extract_frames_from_video(args.video, n_frames=args.n_frames)
    else:
        print(f"Loading images from {args.image_dir} …")
        frames = load_frames_from_dir(args.image_dir)

    print(f"  → {len(frames)} frames ready.\n")

    # ------------------------------------------------------------------
    # Mood prediction
    # ------------------------------------------------------------------
    print("Running mood inference …")
    mood, probs = predict_mood_from_frames(frames, model, class_names, device)
    print(f"Predicted mood: {mood.upper()}")
    print_mood_probabilities(probs, mood)

    # ------------------------------------------------------------------
    # Music recommendations
    # ------------------------------------------------------------------
    try:
        library = load_music_library(args.library)
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        print("Skipping music recommendations.")
        return

    tracks = recommend_tracks(mood, library, top_k=args.top_k)
    print_recommendations(tracks)


if __name__ == "__main__":
    main()
