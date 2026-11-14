"""Draw boxes automatically so you only have to FIX labels instead of drawing them.  (PC)

Uses YOLOE-26, an open-vocabulary YOLO: it finds whatever names are in
config.yaml ("fox", "heron", ...), even animals the normal YOLO model has
never been trained on. It is too big for the UNO Q, but fine on a laptop.

  python prelabel.py                      # label every image that has no label yet
  python prelabel.py --preview            # also save images with boxes drawn, to check quickly
  python prelabel.py --overwrite          # redo everything (throws away your fixes!)
  python prelabel.py --model yolo26s.pt   # use a normal COCO model instead (only COCO animals)

Then open data/raw/images in a labelling tool, load the YOLO labels from
data/raw/labels (with data/raw/labels/classes.txt), fix mistakes, and save back
in YOLO format to the same folder. See README.md.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from common import DATA, RAW_IMAGES, RAW_LABELS, list_images, load_config, write_classes_txt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yoloe-26s-seg.pt", help="yoloe-26[s|m|l]-seg.pt, or any COCO yolo model")
    ap.add_argument("--conf", type=float, default=0.25, help="min confidence; lower = more boxes to delete later")
    ap.add_argument("--overwrite", action="store_true", help="also relabel images that already have labels")
    ap.add_argument("--preview", action="store_true", help="save annotated copies to data/raw/preview")
    args = ap.parse_args()

    classes = load_config()["classes"]
    write_classes_txt(classes)

    todo = [p for p in list_images(RAW_IMAGES) if args.overwrite or not (RAW_LABELS / f"{p.stem}.txt").exists()]
    if not todo:
        print("Nothing to label (all images already have a label file; use --overwrite to redo).")
        return

    model = YOLO(args.model)
    if "yoloe" in args.model.lower():
        model.set_classes(classes)  # the class names ARE the text prompts
    else:
        unknown = [c for c in classes if c not in model.names.values()]
        if unknown:
            print(f"Warning: {args.model} does not know {unknown}; those won't be pre-labelled.")

    preview_dir = DATA / "raw" / "preview"
    if args.preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    counts = {c: 0 for c in classes}
    empty = 0
    for r in model.predict([str(p) for p in todo], conf=args.conf, stream=True, verbose=False):
        lines = []
        for cls, (x, y, w, h) in zip(r.boxes.cls.int().tolist(), r.boxes.xywhn.tolist()):
            name = r.names[cls]
            if name in classes:
                lines.append(f"{classes.index(name)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                counts[name] += 1
        stem = Path(r.path).stem
        (RAW_LABELS / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        empty += not lines
        if args.preview:
            r.save(filename=str(preview_dir / f"{stem}.jpg"), masks=False)

    print(f"Pre-labelled {len(todo)} images: {counts}, {empty} with no animal (kept as background).")
    print("Now check and fix them in a labelling tool - see README.md.")


if __name__ == "__main__":
    main()
