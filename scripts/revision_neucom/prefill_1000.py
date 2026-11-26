"""Seed the jobs of the 1,000-image queue (QUEUE_N=1000 queue_500.py) with the rows of
the 500-image runs, so that only images 500-999 are computed.

Every evaluation is seeded from its own (seed, model, method, image), so the rows of
the first 500 images are exactly those a 1,000-image run would produce. The runner
and the scripts skip rows that are already present:

  grid  revision_grid_1000_shards/<shard>/results.json receives the rows of
        revision_grid_500_shards/<shard>/results.json
  ms    ms_<model>_<methods>_n500_s3_k10.csv -> _n1000_s3_k10.csv
  maps  results/revision_neucom/maps/*.npz are shared; maps.py adds images 500-999
  loc   the dense-mask localisation already covers all 1,000 images
ffe.py has no resume and recomputes its (cheap) rows.

usage (repository root): python scripts/revision_neucom/prefill_1000.py
"""
import glob
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OLD, NEW = os.path.join("results", "revision_grid_500_shards"), os.path.join("results", "revision_grid_1000_shards")
NEU = os.path.join("results", "revision_neucom")


def key(r):
    return r["model"], r["method"], r["image_idx"], r.get("seed", 0)


for src in sorted(glob.glob(os.path.join(OLD, "*", "results.json"))):
    shard = os.path.basename(os.path.dirname(src))
    with open(src, encoding="utf-8") as fh:
        old = [r for r in json.load(fh) if not r.get("error")]
    dst_dir = os.path.join(NEW, shard)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "results.json")
    rows = []
    if os.path.exists(dst):
        try:
            with open(dst, encoding="utf-8") as fh:
                rows = [r for r in json.load(fh) if not r.get("error")]
        except ValueError:
            rows = []          # a write cut short by a killed run; rebuilt below
    seen = {key(r) for r in rows}
    add = [r for r in old if key(r) not in seen]
    rows = sorted(rows + add, key=lambda r: (r["method"], r["image_idx"]))
    with open(dst + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    os.replace(dst + ".tmp", dst)
    print(f"grid {shard}: {len(add)} rows of the 500-image run added, {len(rows)} in total")

for src in sorted(glob.glob(os.path.join(NEU, "ms_*_n500_s3_k10.csv"))):
    dst = src.replace("_n500_s3_k10.csv", "_n1000_s3_k10.csv")
    if os.path.exists(dst):
        print("  kept existing", os.path.basename(dst))
    else:
        shutil.copy2(src, dst)
        print("  copied", os.path.basename(src), "->", os.path.basename(dst))
