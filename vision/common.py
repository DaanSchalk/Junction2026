"""Shared paths and config loading. Nothing interesting happens here."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config.yaml"

DATA = ROOT / "data"
RAW_IMAGES = DATA / "raw" / "images"   # everything capture.py / get_example_images.py saves
RAW_LABELS = DATA / "raw" / "labels"   # one YOLO .txt per image (empty file = no animal)
DATASET = DATA / "dataset"             # train/val split, rebuilt by split_dataset.py
MODELS = ROOT / "models"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        cfg = yaml.safe_load(f)
    if not cfg.get("classes"):
        raise SystemExit("config.yaml: 'classes' is empty")
    return cfg


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTS)


def burst_id(image: Path) -> str:
    """'20261114-105301_b0007_f2.jpg' -> '20261114-105301_b0007'.

    Frames from the same burst are near-duplicates, so they must end up in the
    same split (train OR val), otherwise validation scores look too good.
    """
    stem = image.stem
    return stem.rsplit("_f", 1)[0] if "_f" in stem else stem


def write_classes_txt(classes: list[str]) -> None:
    """Labelling tools (X-AnyLabeling, labelImg, ...) want a classes.txt next to the labels."""
    RAW_LABELS.mkdir(parents=True, exist_ok=True)
    (RAW_LABELS / "classes.txt").write_text("\n".join(classes) + "\n")
