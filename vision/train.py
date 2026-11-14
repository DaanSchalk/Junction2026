"""Fine-tune YOLO26n on your animals and export it for the UNO Q.  (PC)

Run split_dataset.py first. Training on a laptop CPU works for small datasets
(minutes to an hour); an NVIDIA GPU or Google Colab is much faster.
Do NOT train on the UNO Q itself.

  python train.py                       # 100 epochs, then export to NCNN
  python train.py --epochs 3            # quick smoke test
  python train.py --keep best           # ship the best-scoring epoch instead of the last one
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
    ap.add_argument("--patience", type=int, default=None,
                    help="stop early after this many epochs without improvement (default: 30, "
                         "or never when the validation set is small)")
    ap.add_argument("--keep", choices=["auto", "best", "last"], default="auto",
                    help="which weights to ship: the best-scoring epoch or the last one "
                         "(auto: last when the validation set is small)")
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

    # With only a few validation photos the score per epoch is mostly noise: "best epoch" and early
    # stopping then pick an under-trained model. (Seen on the first 66 photos of the minis: the
    # "best" epoch was 14 and missed the held-out fox; the last epoch found all four animals.)
    # So on small validation sets: train the full schedule and keep the last weights.
    val_boxes = sum(len([r for r in f.read_text().splitlines() if r.strip()])
                    for f in (DATASET / "labels" / "val").glob("*.txt"))
    small_val = val_boxes < 100
    patience = args.patience if args.patience is not None else (0 if small_val else 30)
    keep = args.keep if args.keep != "auto" else ("last" if small_val else "best")
    if small_val:
        print(f"Small validation set ({val_boxes} boxes): "
              f"{'no early stopping' if patience == 0 else f'patience {patience}'}, keeping the {keep} weights.")

    model = YOLO(cfg["base_model"])
    model.train(
        data=str(data_yaml),
        imgsz=cfg["imgsz"],
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        patience=patience if patience > 0 else args.epochs,  # patience >= epochs = never stop early
        project=str(ROOT / "runs"),
        name="animals",
        # Ultralytics' default augmentations (flips, colour, scale, mosaic) are fine here.
    )

    chosen = Path(model.trainer.best if keep == "best" else model.trainer.last)
    target = MODELS / "animals.pt"
    shutil.copy2(chosen, target)
    print(f"\nWeights ({keep} epoch): {chosen} -> {target}")
    print(f"Training plots: {model.trainer.save_dir}")
    export(target, cfg["imgsz"])


if __name__ == "__main__":
    main()
