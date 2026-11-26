"""Join the faithfulness rows of the 500-image run with those of images 500-999.

ffe.py ... 1000 METHOD 500 writes ffe_<model>_<method>_n1000_from500.csv per method
(queue_500.py with QUEUE_N=1000); this script writes the file a single 1,000-image run
would have written, ffe_<model>_<methods>_n1000.csv, from
ffe_<model>_<methods>_n500.csv and those parts, and checks that every
(method, protocol, image) row is present exactly once.

usage (repository root): python scripts/revision_neucom/merge_ffe.py
"""
import csv
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "revision_neucom")
OLD, NEW, FROM = 500, 1000, 500


def read(fp):
    with open(fp, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


for old_fp in sorted(glob.glob(os.path.join(OUT, f"ffe_*_n{OLD}.csv"))):
    stem = os.path.basename(old_fp)[: -len(f"_n{OLD}.csv")]          # ffe_<model>_<methods>
    old = read(old_fp)
    model = old[0]["model"]
    methods = list(dict.fromkeys(r["method"] for r in old))
    parts = [os.path.join(OUT, f"ffe_{model}_{m}_n{NEW}_from{FROM}.csv") for m in methods]
    missing = [p for p in parts if not os.path.exists(p)]
    if missing:
        print(f"skip {stem}: missing {', '.join(os.path.basename(p) for p in missing)}")
        continue
    rows = [r for r in old if int(r["idx"]) < FROM]
    for p in parts:
        rows += read(p)
    keys = [(r["method"], r["protocol"], int(r["idx"])) for r in rows]
    expect = {(m, p, i) for m in methods for p in ("abs", "pos", "signed") for i in range(NEW)}
    assert len(keys) == len(set(keys)), f"{stem}: duplicate rows"
    assert set(keys) == expect, f"{stem}: {len(expect - set(keys))} rows missing"
    order = {m: k for k, m in enumerate(methods)}
    proto = {"abs": 0, "pos": 1, "signed": 2}
    rows.sort(key=lambda r: (order[r["method"]], proto[r["protocol"]], int(r["idx"])))
    out = os.path.join(OUT, f"{stem}_n{NEW}.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "method", "protocol", "idx", "fc", "fe"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.basename(out)}: {len(rows)} rows")
