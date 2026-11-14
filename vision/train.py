"""Fine-tune YOLO26n on your animals and export it for the UNO Q.  (PC)

Run split_dataset.py first. Training on a laptop CPU works for small datasets
(minutes to an hour); an NVIDIA GPU or Google Colab is much faster.
Do NOT train on the UNO Q itself.

  python train.py                       # 100 epochs, then export to NCNN
  python train.py --epochs 3            # quick smoke test
  python train.py --device 0            # use NVIDIA GPU 0 ("mps" on Apple silicon)
  python train.py --export-only models/animals.pt

Result:
  models/animals.pt            PyTorch weights (for the PC / further training)
  models/animals_ncnn_model/   fast ARM format -> copy this folder to the UNO Q
  runs/animals*/               training curves, confusion matrix, example predictions
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from common import DATASET, MODELS, ROOT, load_config


def export(weights: Path, imgsz: int) -> Path:
    out = YOLO(str(weights)).export(format="ncnn", imgsz=imgsz)
    print(f"\nModel for the UNO Q: {out}\nCopy that whole folder to the board (same relative path).")
    return Path(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16, help="lower if you run out of memory")
    ap.add_argument("--device", default=None, help="'cpu', '0' for GPU 0, 'mps' for Apple; default: auto")
    ap.add_argument("--patience", type=int, default=30, help="stop early after this many epochs without improvement")
    ap.add_argument("--export-only", type=Path, help="skip training, just export these weights")
    args = ap.parse_args()

    cfg = load_config()
    MODELS.mkdir(exist_ok=True)

    if args.export_only:
        export(args.export_only, cfg["imgsz"])
        return

    data_yaml = DATASET / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit("No dataset yet - run split_dataset.py first.")

    model = YOLO(cfg["base_model"])
    model.train(
        data=str(data_yaml),
        imgsz=cfg["imgsz"],
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=str(ROOT / "runs"),
        name="animals",
        # Ultralytics' default augmentations (flips, colour, scale, mosaic) are fine here.
    )

    best = Path(model.trainer.best)
    target = MODELS / "animals.pt"
    shutil.copy2(best, target)
    print(f"\nBest weights: {best} -> {target}")
    print(f"Training plots: {model.trainer.save_dir}")
    export(target, cfg["imgsz"])


if __name__ == "__main__":
    main()
