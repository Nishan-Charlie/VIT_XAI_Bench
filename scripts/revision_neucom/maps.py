"""Signed attribution maps for the faithfulness study (ffe.py).

ffe.py scores the signed maps of the inherently signed methods under three
protocols. loc.py keeps them for the first SAVE_MAPS_N images as a by-product of
the localisation run; this script computes any that are missing (methods that
loc.py ran on a larger image set in an earlier call, or rows written before a
crash), exactly as loc.py does: ground-truth target, seeded with
evaluation_seed(0, model, method, image).

usage: python maps.py MODEL N METHOD[,METHOD...]
AttnLRP must run alone in its own process (class-level patches).
"""
import os
import sys
import time

import numpy as np
import torch
from common import (
    DEVICE,
    METHODS,
    OUT,
    SIGNED_METHODS,
    evaluation_seed,
    load_data,
    load_model,
    record_failure,
    set_seed,
    signed_attribution,
)

model_name, N = sys.argv[1], int(sys.argv[2])
methods = sys.argv[3].split(",")
if "attnlrp" in methods and len(methods) > 1:
    raise SystemExit("attnlrp must run in its own process")
unsigned = [m for m in methods if m not in SIGNED_METHODS]
if unsigned:
    raise SystemExit(f"only the signed methods need saved maps, not {unsigned}")

data = load_data(N)
mw = load_model(model_name)
is_vit = mw.cam_format in ("tokens_cls", "tokens_nocls")
os.makedirs(os.path.join(OUT, "maps"), exist_ok=True)


def save(fp, maps):
    tmp = fp[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **maps)
    os.replace(tmp, fp)


for meth in methods:
    if meth == "attnlrp" and not is_vit:
        continue
    fp = os.path.join(OUT, "maps", f"{model_name}__{meth}.npz")
    maps = {}
    if os.path.exists(fp):
        with np.load(fp) as z:
            maps = {k: z[k] for k in z.files}
    todo = [d for d in data if str(d["idx"]) not in maps]
    print(f"{model_name} {meth}: {len(maps)} maps present, {len(todo)} to compute", flush=True)
    reg = METHODS.get(meth)(model_wrapper=mw, model=mw.model) if meth == "attnlrp" else None
    t_start = time.time()
    for k, d in enumerate(todo):
        set_seed(evaluation_seed(0, model_name, meth, d["idx"]))
        x = d["image"].unsqueeze(0).to(DEVICE)
        try:
            a = signed_attribution(meth, mw, x, d["label"], registry_fn=reg).float()
        except RuntimeError as e:
            if "CUDA error" in str(e) or "illegal" in str(e) or "CUBLAS" in str(e):
                save(fp, maps)
                raise SystemExit(f"CUDA fault, exiting for a clean retry: {str(e)[:120]}") from e
            print(f"  FAIL {meth} idx={d['idx']}: {str(e)[:120]}", flush=True)
            record_failure("maps", model_name, meth, d["idx"], e)
            torch.cuda.empty_cache()
            continue
        maps[str(d["idx"])] = a.cpu().numpy().astype(np.float32)
        if k % 50 == 49:
            save(fp, maps)
            print(f"{model_name} {meth} {k + 1}/{len(todo)} ({time.time() - t_start:.0f}s)", flush=True)
    if todo:
        save(fp, maps)
    torch.cuda.empty_cache()
print("DONE maps", model_name, ",".join(methods))
