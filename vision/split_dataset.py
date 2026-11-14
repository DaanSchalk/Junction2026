"""Turn data/raw into the folder layout YOLO trains on.  (PC)

  data/dataset/
    images/train/*.jpg   labels/train/*.txt
    images/val/*.jpg     labels/val/*.txt
    data.yaml            <- what train.py points YOLO at

Only images WITH a label file are used (an empty .txt = "no animal here",
which is useful background). Images without a label file are treated as
"not labelled yet" and skipped.

Whole bursts go to either train or val, never split up, because frames of one
burst are almost identical and would make validation look better than it is.
Validation bursts are picked per animal (and per "no animal"), so even a small
dataset is tested on every class.

  python split_dataset.py              # 80/20 split
  python split_dataset.py --val 0.15
"""

import argparse
import random
import shutil
from collections import Counter, defaultdict

import yaml

from common import DATASET, RAW_IMAGES, RAW_LABELS, burst_id, list_images, load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=float, default=0.2, help="fraction of images for validation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    classes = load_config()["classes"]
    images = list_images(RAW_IMAGES)
    labelled = [p for p in images if (RAW_LABELS / f"{p.stem}.txt").exists()]
    if len(images) != len(labelled):
        print(f"Skipping {len(images) - len(labelled)} images without a label file (not labelled yet).")
    if not labelled:
        raise SystemExit("No labelled images. Run prelabel.py first.")

    groups = defaultdict(list)
    for p in labelled:
        groups[burst_id(p)].append(p)
    if len(groups) < 2:
        raise SystemExit("Need photos from at least 2 bursts to make a train and a val set.")

    def burst_class(paths) -> str:
        """The animal a burst shows (its most common label), or 'background' if it shows none."""
        ids = Counter(int(r.split()[0]) for p in paths
                      for r in (RAW_LABELS / f"{p.stem}.txt").read_text().splitlines() if r.strip())
        cid = ids.most_common(1)[0][0] if ids else None
        return "background" if cid is None else classes[cid] if cid < len(classes) else f"class {cid}"

    by_class = defaultdict(list)
    for k in sorted(groups):
        by_class[burst_class(groups[k])].append(k)

    # Per class: hold out a share of its bursts, at least one when it has two or more.
    rng = random.Random(args.seed)
    val_bursts = set()
    for name in sorted(by_class):
        keys = by_class[name]
        rng.shuffle(keys)
        if len(keys) < 2:
            print(f"Warning: only one burst of '{name}', so it can't be validated. Take more photos of it.")
            continue
        val_bursts.update(keys[:max(1, round(len(keys) * args.val))])
    split = {p: "val" if k in val_bursts else "train" for k in groups for p in groups[k]}

    if DATASET.exists():
        shutil.rmtree(DATASET)
    stats = {s: Counter() for s in ("train", "val")}
    for p, s in split.items():
        for kind in ("images", "labels"):
            (DATASET / kind / s).mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, DATASET / "images" / s / p.name)
        lbl = RAW_LABELS / f"{p.stem}.txt"
        shutil.copy2(lbl, DATASET / "labels" / s / lbl.name)
        rows = [r for r in lbl.read_text().splitlines() if r.strip()]
        for r in rows:
            cid = int(r.split()[0])
            if cid >= len(classes):
                raise SystemExit(f"{lbl.name}: class id {cid} but config.yaml only has {len(classes)} classes")
            stats[s][classes[cid]] += 1
        stats[s]["images"] += 1
        stats[s]["background images"] += not rows

    (DATASET / "data.yaml").write_text(yaml.safe_dump({
        "path": str(DATASET.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(classes)),
    }, sort_keys=False))

    for s in ("train", "val"):
        print(f"{s:5}: " + ", ".join(f"{k}={v}" for k, v in stats[s].items()))
    for c in classes:
        if stats["val"][c] == 0:
            print(f"Warning: no '{c}' in the validation set - collect more photos of it.")
    print(f"Wrote {DATASET / 'data.yaml'}")


if __name__ == "__main__":
    main()
