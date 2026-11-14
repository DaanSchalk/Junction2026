"""Grab a few public example photos so you can try the whole pipeline before you
have your own camera shots.

Downloads Ultralytics' tiny COCO128 sample (~7 MB) and copies the photos that
contain any animal from config.yaml into data/raw/images. This is only for
testing the pipeline: ~15 photos will not give a good model.

  python get_example_images.py                # images only -> label them with prelabel.py
  python get_example_images.py --with-labels  # also copy COCO's own boxes (skip prelabel)
"""

import argparse
import shutil
from pathlib import Path

from ultralytics.data.utils import check_det_dataset

from common import RAW_IMAGES, RAW_LABELS, load_config, write_classes_txt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-labels", action="store_true", help="also write the COCO ground-truth boxes")
    args = ap.parse_args()

    classes = load_config()["classes"]
    data = check_det_dataset("coco128.yaml")  # downloads on first use
    coco_names = data["names"]                # {0: 'person', ..., 16: 'dog', ...}
    coco_to_ours = {i: classes.index(n) for i, n in coco_names.items() if n in classes}
    missing = [c for c in classes if c not in coco_names.values()]
    if missing:
        print(f"Note: COCO has no examples of {missing}; you'll need your own photos for those.")

    img_dir = Path(data["train"])
    lbl_dir = img_dir.parent.parent / "labels" / img_dir.name
    RAW_IMAGES.mkdir(parents=True, exist_ok=True)
    RAW_LABELS.mkdir(parents=True, exist_ok=True)

    n = 0
    for lbl in sorted(lbl_dir.glob("*.txt")):
        rows = [r.split() for r in lbl.read_text().splitlines() if r.strip()]
        ours = [[str(coco_to_ours[int(r[0])])] + r[1:] for r in rows if int(r[0]) in coco_to_ours]
        if not ours:
            continue
        n += 1
        name = f"example_b{n:04d}_f0"
        shutil.copy(img_dir / f"{lbl.stem}.jpg", RAW_IMAGES / f"{name}.jpg")
        if args.with_labels:
            (RAW_LABELS / f"{name}.txt").write_text("\n".join(" ".join(r) for r in ours) + "\n")

    write_classes_txt(classes)
    print(f"Copied {n} example photos to {RAW_IMAGES}" + (" (with labels)" if args.with_labels else ""))


if __name__ == "__main__":
    main()
