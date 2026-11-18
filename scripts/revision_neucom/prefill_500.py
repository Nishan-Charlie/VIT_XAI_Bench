"""Seed the jobs of queue_500.py with the rows of the earlier revision runs, so that
only the images those runs did not cover are computed.

The earlier runs covered the first 100 images (grid: RN-50 and ViT-B/16; dense-mask
localisation of IG, RISE, LIME and Occlusion; multi-seed MS), the first 50 (grid:
the other six backbones) and the first 30 (multi-seed MS of VarGrad). Every
evaluation is seeded from its own (seed, model, method, image), so the remaining
images are computed exactly as if the earlier runs had covered them. The runner
and the scripts skip rows that are already present:

  grid  the shard's results.json receives the earlier rows of its backbone and
        methods
  loc   loc_<model>_<methods>_n100.csv   -> _n1000.csv
  ms    ms_<model>_<methods>_n100|n30_*  -> _n500_*
  maps  the signed maps of the first 100 images (loc.py merges new ones into them)
ffe.py has no resume and recomputes its (cheap) rows.

usage: python prefill_500.py OLD_REPO TARGET_REPO MODEL[,MODEL...]
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from queue_500 import CHEAP, EXTRA, GROUPS, N_GRID, SHARDS, VIT  # noqa: E402

old_root, target, models = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
OLD_R = os.path.join(old_root, "results")
OLD_N = os.path.join(OLD_R, "revision_neucom")
NEW_N = os.path.join(target, "results", "revision_neucom")
os.makedirs(os.path.join(NEW_N, "maps"), exist_ok=True)

old_rows = []
for run in ("revision_grid_seeded", "revision_grid_seeded_50", "revision_grid_seeded_attnlrp"):
    with open(os.path.join(OLD_R, run, "results.json"), encoding="utf-8") as fh:
        old_rows += [r for r in json.load(fh) if not r.get("error")]


def prefill_shard(model, group, methods):
    run_dir = os.path.join(target, SHARDS, f"{model}__{group}")
    os.makedirs(run_dir, exist_ok=True)
    fp = os.path.join(run_dir, "results.json")
    rows = []
    if os.path.exists(fp):
        try:
            with open(fp, encoding="utf-8") as fh:
                rows = [r for r in json.load(fh) if not r.get("error")]
        except ValueError:
            rows = []          # a write cut short by a killed run; rebuilt below
    seen = {(r["model"], r["method"], r["image_idx"], r.get("seed", 0)) for r in rows}
    add = [r for r in old_rows if r["model"] == model and r["method"] in methods and r["image_idx"] < N_GRID
           and (r["model"], r["method"], r["image_idx"], r.get("seed", 0)) not in seen]
    rows = sorted(rows + add, key=lambda r: (r["method"], r["image_idx"]))
    with open(fp + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    os.replace(fp + ".tmp", fp)
    print(f"grid {model}__{group}: {len(add)} earlier rows added, {len(rows)} in total")


def copy(src, dst):
    s, d = os.path.join(OLD_N, src), os.path.join(NEW_N, dst)
    if not os.path.exists(s):
        print("  (no earlier file)", src)
    elif os.path.exists(d):
        print("  kept existing", dst)
    else:
        shutil.copy2(s, d)
        print("  copied", src, "->", dst)


for m in models:
    if m in EXTRA:
        copy(f"ms_{m}_grad_cam_grad_cam_plus_plus_grad_cam_plus_plus_stab_n100_s3_k10.csv",
             f"ms_{m}_grad_cam_grad_cam_plus_plus_grad_cam_plus_plus_stab_n500_s3_k10.csv")
        continue
    for g, meths in GROUPS.items():
        prefill_shard(m, g, meths)
    for x in ("integrated_gradients_rise", "lime", "occlusion"):
        copy(f"loc_{m}_{x}_n100.csv", f"loc_{m}_{x}_n1000.csv")
    cheap = CHEAP.replace(",", "_")
    copy(f"ms_{m}_{cheap}_n100_s3_k10.csv", f"ms_{m}_{cheap}_n500_s3_k10.csv")
    copy(f"ms_{m}_vargrad_n30_s3_k10.csv", f"ms_{m}_vargrad_n500_s3_k10.csv")
    for meth in ("integrated_gradients", "input_x_gradient", "gradient_shap", "occlusion", "lime"):
        copy(os.path.join("maps", f"{m}__{meth}.npz"), os.path.join("maps", f"{m}__{meth}.npz"))
    if m == VIT:
        prefill_shard(m, "attnlrp", ["attnlrp"])
        copy(f"ms_{m}_attnlrp_n100_s3_k10.csv", f"ms_{m}_attnlrp_n500_s3_k10.csv")
        copy(os.path.join("maps", f"{m}__attnlrp.npz"), os.path.join("maps", f"{m}__attnlrp.npz"))
