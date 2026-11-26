"""Job queue for the 500-image extension of the revision analyses on a multi-GPU machine.

Jobs (image counts as in the revised paper):
  grid   the five-metric grid, configs/revision_grid_500{,_attnlrp}.yaml, split into
         one run per (backbone, method group) under results/revision_grid_500_shards/
  loc    dense-mask localisation of IG, RISE, LIME and Occlusion on all 1,000 images
         (the other methods already cover them); keeps the signed maps of the first
         500 images for the faithfulness study
  maps   any signed map that loc did not keep (input x gradient, GradientSHAP, AttnLRP,
         rows written before a crash)
  ffe    faithfulness protocols (abs / pos / signed) on the first 500 images
  ms     multi-seed Max-Sensitivity (3 seeds, 10 perturbations) on the first 500
         images: the eight main backbones and the thirteen of the capacity study

Every evaluation is seeded from its own (seed, backbone, method, image), so a job
gives the same rows whichever worker or GPU runs it, and a killed job resumes.

Workers are bound to one GPU each and repeatedly claim the costliest runnable job
(an atomic mkdir under results/queue_500/claims), run it with up to three attempts,
and mark it done. Several workers may share a GPU.

usage (from the repository root):
  python scripts/revision_neucom/queue_500.py worker --gpu 0 --name g0a [--threads 4]
  python scripts/revision_neucom/queue_500.py status
  python scripts/revision_neucom/queue_500.py reset     # drop claims of dead workers
  python scripts/revision_neucom/queue_500.py reserve --jobs REGEX --owner NAME
      # mark jobs as run elsewhere (another machine with its own queue directory,
      # whose worker is started with the same --jobs REGEX); workers here skip them
  touch results/queue_500/STOP                           # workers stop after their job
"""
import argparse
import csv
import json
import os
import re
import socket
import subprocess
import sys
import time

import psutil
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
# QUEUE_N=1000 extends the grid, faithfulness and multi-seed analyses to all 1,000
# images (queue results/queue_1000, configs/revision_grid_1000*.yaml); seed it from
# the 500-image runs with prefill_1000.py so that only images 500-999 are computed.
N = int(os.environ.get("QUEUE_N", "500"))
Q = os.path.join("results", f"queue_{N}")
SHARDS = os.path.join("results", f"revision_grid_{N}_shards")
FAILURES_CSV = os.path.join("results", "revision_neucom", "failures.csv")
N_GRID, N_LOC, N_FFE, N_MS = N, 1000, N, N
# ffe.py has no resume: when extending the 500-image run, score only images 500..N-1,
# one job per (backbone, method), and join them with the 500-image rows (merge_ffe.py)
FFE_FROM = 500 if N > 500 else 0

MAIN = ["resnet50", "vit_base_patch16_224", "swin_base_patch4_window7_224", "pvt_v2_b2",
        "maxvit_small_tf_224", "mobilevitv2_100", "efficientvit_b1", "efficientvit_b2"]
EXTRA = ["convnext_tiny", "convnext_base", "vit_small_patch16_224", "deit_small_patch16_224",
         "deit_base_patch16_224", "swin_tiny_patch4_window7_224", "maxvit_tiny_tf_224",
         "mobilevitv2_200", "efficientvit_b3", "resnet18", "resnet101", "pvt_v2_b0", "pvt_v2_b4"]
VIT = "vit_base_patch16_224"
GROUPS = {"gradA": ["saliency", "integrated_gradients", "smoothgrad", "input_x_gradient", "attention_rollout"],
          "gradB": ["vargrad", "gradient_shap", "grad_cam", "grad_cam_plus_plus"],
          "lime_rise": ["lime", "rise"],
          "occlusion": ["occlusion"]}
SIGNED = "integrated_gradients,input_x_gradient,gradient_shap,occlusion,lime"
CHEAP = "saliency,input_x_gradient,grad_cam,grad_cam_plus_plus,grad_cam_plus_plus_stab,attention_rollout"

# Rough cost in hours on the laptop GPU of the earlier runs; only orders the queue
# (costliest first) so that the long jobs do not start last.
GRID_H = {"resnet50": (1.4, 1.4, 1.6, 2.6), "vit_base_patch16_224": (2.5, 2.5, 2.6, 4.1),
          "swin_base_patch4_window7_224": (6.2, 6.3, 6.4, 11.8), "pvt_v2_b2": (2.7, 2.8, 3.0, 5.0),
          "maxvit_small_tf_224": (11.3, 11.4, 9.0, 13.3), "mobilevitv2_100": (1.8, 1.8, 2.0, 3.3),
          "efficientvit_b1": (2.2, 2.2, 2.5, 4.1), "efficientvit_b2": (2.9, 2.9, 3.2, 5.3)}
OCC_H = {"resnet50": 1.6, "vit_base_patch16_224": 2.0, "swin_base_patch4_window7_224": 5.2, "pvt_v2_b2": 3.0,
         "maxvit_small_tf_224": 6.7, "mobilevitv2_100": 2.0, "efficientvit_b1": 2.6, "efficientvit_b2": 3.3}
VARGRAD_H = {"maxvit_small_tf_224": 14.8, "swin_base_patch4_window7_224": 1.2, "vit_base_patch16_224": 1.1}


def py(*args):
    return [sys.executable, *map(str, args)]


def shard_config(model, group, methods, src):
    """Write the config of one grid shard: the canonical sweep restricted to one
    backbone and one method group, in its own run directory."""
    with open(src, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["models"] = [model]
    cfg["methods"] = [m for m in cfg["methods"] if m in methods]
    cfg["output_dir"] = SHARDS
    cfg["run_name"] = f"{model}__{group}"
    path = os.path.join(Q, "configs", f"{model}__{group}.yaml")
    text = f"# generated by queue_500.py from {src}\n" + yaml.safe_dump(cfg, sort_keys=False)
    old = None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            old = fh.read()
    if old != text:           # a running worker may be reading it: rewrite only on change
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f".{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    return path


def jobs():
    J = []

    def add(name, cmd, cost, deps=(), run_dir=None):
        J.append(dict(name=name, cmd=cmd, cost=cost, deps=list(deps), run_dir=run_dir,
                      script=name.split(":")[0], model=name.split(":")[1]))

    for m in MAIN:
        for (g, meths), h in zip(GROUPS.items(), GRID_H[m], strict=True):
            cfg = shard_config(m, g, meths, f"configs/revision_grid_{N}.yaml")
            add(f"grid:{m}:{g}", py("-m", "xai_bench.runner", "--config", cfg), h,
                run_dir=os.path.join(SHARDS, f"{m}__{g}"))
        locs = [f"loc:{m}:integrated_gradients,rise", f"loc:{m}:lime", f"loc:{m}:occlusion"]
        add(locs[0], py("scripts/revision_neucom/loc.py", m, N_LOC, "integrated_gradients,rise"), 1.5)
        add(locs[1], py("scripts/revision_neucom/loc.py", m, N_LOC, "lime"), 0.5)
        add(locs[2], py("scripts/revision_neucom/loc.py", m, N_LOC, "occlusion"), OCC_H[m])
        add(f"maps:{m}", py("scripts/revision_neucom/maps.py", m, N_FFE, SIGNED), 0.3, deps=locs)
        if FFE_FROM:
            for meth in SIGNED.split(","):
                add(f"ffe:{m}:{meth}", py("scripts/revision_neucom/ffe.py", m, N_FFE, meth, FFE_FROM), 0.4,
                    deps=[f"maps:{m}"])
        else:
            add(f"ffe:{m}", py("scripts/revision_neucom/ffe.py", m, N_FFE, SIGNED), 1.0, deps=[f"maps:{m}"])
        add(f"ms:{m}:cheap", py("scripts/revision_neucom/ms.py", m, N_MS, "0,1,2", 10, CHEAP), 1.0)
        add(f"ms:{m}:vargrad", py("scripts/revision_neucom/ms.py", m, N_MS, "0,1,2", 10, "vargrad"),
            VARGRAD_H.get(m, 0.4))
    cfg = shard_config(VIT, "attnlrp", ["attnlrp"], f"configs/revision_grid_{N}_attnlrp.yaml")
    add(f"grid:{VIT}:attnlrp", py("-m", "xai_bench.runner", "--config", cfg), 3.0,
        run_dir=os.path.join(SHARDS, f"{VIT}__attnlrp"))
    add(f"maps:{VIT}:attnlrp", py("scripts/revision_neucom/maps.py", VIT, N_FFE, "attnlrp"), 0.3)
    add(f"ffe:{VIT}:attnlrp", py("scripts/revision_neucom/ffe.py", VIT, N_FFE, "attnlrp",
                                 *([FFE_FROM] if FFE_FROM else [])), 0.3, deps=[f"maps:{VIT}:attnlrp"])
    add(f"ms:{VIT}:attnlrp", py("scripts/revision_neucom/ms.py", VIT, N_MS, "0,1,2", 10, "attnlrp"), 2.0)
    for m in EXTRA:
        add(f"ms:{m}:gcpp", py("scripts/revision_neucom/ms.py", m, N_MS, "0,1,2", 10,
                               "grad_cam,grad_cam_plus_plus,grad_cam_plus_plus_stab"), 0.4)
    return J


def key(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def path(kind, name):
    return os.path.join(Q, kind, key(name))


def is_done(name):
    return os.path.exists(path("done", name))


def is_failed(name):
    return os.path.exists(path("failed", name))


def is_claimed(name):
    return os.path.isdir(path("claims", name))


def failures_lines(job):
    """Rows of results/revision_neucom/failures.csv written by this job's script for
    this job's backbone (other jobs append to the same file)."""
    if not os.path.exists(FAILURES_CSV):
        return 0
    with open(FAILURES_CSV, encoding="utf-8", errors="replace", newline="") as fh:
        return sum(1 for r in csv.reader(fh) if len(r) > 1 and r[0] == job["script"] and r[1] == job["model"])


def attempt_ok(job, before_failures):
    """A job succeeded when it exited 0 and left no failed evaluation behind: the
    runner lists them in failures.json, the revision scripts in failures.csv."""
    if job["run_dir"]:
        fp = os.path.join(job["run_dir"], "failures.json")
        if not os.path.exists(fp):
            return False
        with open(fp, encoding="utf-8") as fh:
            f = json.load(fh)
        return not f.get("evaluations") and not f.get("method_init")
    return failures_lines(job) == before_failures


def run_job(job, gpu, threads, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), SAVE_MAPS_N=str(N_FFE),
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1",
               TQDM_MININTERVAL="60", TF_CPP_MIN_LOG_LEVEL="3")
    for attempt in range(1, 4):
        before = failures_lines(job)
        t0 = time.time()
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"\n=== attempt {attempt} on gpu {gpu}, {time.strftime('%Y-%m-%d %H:%M:%S')}: "
                     f"{' '.join(job['cmd'])}\n")
            fh.flush()
            if os.name == "nt":   # below-normal priority on Windows
                rc = subprocess.call(job["cmd"], stdout=fh, stderr=subprocess.STDOUT, env=env,
                                     creationflags=0x00004000)
            else:
                rc = subprocess.call(job["cmd"], stdout=fh, stderr=subprocess.STDOUT, env=env,
                                     preexec_fn=lambda: os.nice(10))
        if rc == 0 and attempt_ok(job, before):
            return True, time.time() - t0, attempt
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"=== attempt {attempt} not clean (exit {rc}); retrying in 30 s\n")
        time.sleep(30)
    return False, time.time() - t0, 3


def worker(args):
    for d in ("claims", "done", "failed", "logs"):
        os.makedirs(os.path.join(Q, d), exist_ok=True)
    me = dict(host=socket.gethostname(), pid=os.getpid(), worker=args.name, gpu=args.gpu)
    while True:
        if os.path.exists(os.path.join(Q, "STOP")):
            print(f"[{args.name}] STOP file found, exiting", flush=True)
            return
        J = [j for j in jobs() if not args.jobs or re.search(args.jobs, j["name"])]
        open_jobs = [j for j in J if not is_done(j["name"]) and not is_failed(j["name"])]
        if not open_jobs:
            print(f"[{args.name}] no jobs left, exiting", flush=True)
            return
        runnable = [j for j in open_jobs if not is_claimed(j["name"])
                    and all(is_done(d) for d in j["deps"])]
        job = None
        for j in sorted(runnable, key=lambda j: -j["cost"]):
            try:
                os.mkdir(path("claims", j["name"]))
            except FileExistsError:
                continue
            with open(os.path.join(path("claims", j["name"]), "owner.json"), "w") as fh:
                json.dump(dict(me, job=j["name"], start=time.time()), fh)
            job = j
            break
        if job is None:
            if not any(is_claimed(j["name"]) for j in open_jobs):
                print(f"[{args.name}] the remaining jobs wait on failed jobs, exiting", flush=True)
                return
            time.sleep(60)
            continue
        print(f"[{args.name}] {time.strftime('%H:%M:%S')} start {job['name']}", flush=True)
        ok, dt, attempts = run_job(job, args.gpu, args.threads, path("logs", job["name"]) + ".log")
        with open(path("done" if ok else "failed", job["name"]), "w") as fh:
            json.dump(dict(me, job=job["name"], seconds=round(dt), attempts=attempts), fh)
        os.rename(path("claims", job["name"]), path("claims", job["name"]) + f".released.{int(time.time())}")
        print(f"[{args.name}] {time.strftime('%H:%M:%S')} {'done' if ok else 'FAILED'} {job['name']} "
              f"({dt / 3600:.2f} h, {attempts} attempt(s))", flush=True)


def rows_so_far(run_dir):
    fp = os.path.join(run_dir, "results.json")
    try:
        with open(fp, encoding="utf-8") as fh:
            return len(json.load(fh))
    except (OSError, ValueError):
        return 0


def status(_args):
    J = jobs()
    done = [j for j in J if is_done(j["name"])]
    failed = [j for j in J if is_failed(j["name"])]
    running = [j for j in J if is_claimed(j["name"]) and j not in done and j not in failed]
    print(f"{len(done)} done, {len(running)} running, {len(failed)} failed, "
          f"{len(J) - len(done) - len(running) - len(failed)} waiting, of {len(J)} jobs")
    for j in running:
        with open(os.path.join(path("claims", j["name"]), "owner.json")) as fh:
            o = json.load(fh)
        extra = f", {rows_so_far(j['run_dir'])} rows" if j["run_dir"] else ""
        print(f"  running  {j['name']:55s} gpu {o['gpu']} {o['worker']:5s} "
              f"{(time.time() - o['start']) / 3600:5.2f} h{extra}")
    for j in failed:
        print(f"  FAILED   {j['name']}")
    hours = 0.0
    for j in done:
        with open(path("done", j["name"])) as fh:
            hours += json.load(fh)["seconds"] / 3600
    print(f"GPU-hours spent on finished jobs: {hours:.1f}")


def reset(_args):
    host = socket.gethostname()
    cdir = os.path.join(Q, "claims")
    for c in sorted(os.listdir(cdir)) if os.path.isdir(cdir) else []:
        if ".released." in c:
            continue
        try:
            with open(os.path.join(cdir, c, "owner.json")) as fh:
                o = json.load(fh)
        except OSError:
            o = {}
        if o.get("host") != host:
            continue              # reserved for, or claimed on, another machine
        if o.get("pid") and psutil.pid_exists(o["pid"]):
            continue              # worker alive: keep the claim (os.kill(pid, 0) would
                                  # terminate the process on Windows)
        os.rename(os.path.join(cdir, c), os.path.join(cdir, c + f".released.{int(time.time())}"))
        print("released", c)


def reserve(args):
    os.makedirs(os.path.join(Q, "claims"), exist_ok=True)
    for j in jobs():
        if re.search(args.jobs, j["name"]) and not is_done(j["name"]):
            try:
                os.mkdir(path("claims", j["name"]))
            except FileExistsError:
                print("already claimed", j["name"])
                continue
            with open(os.path.join(path("claims", j["name"]), "owner.json"), "w") as fh:
                json.dump(dict(host=args.owner, pid=None, worker=args.owner, gpu="-", job=j["name"],
                               start=time.time()), fh)
            print("reserved", j["name"], "for", args.owner)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--gpu", type=int, required=True)
    w.add_argument("--name", required=True)
    w.add_argument("--threads", type=int, default=4)
    w.add_argument("--jobs", default="", help="regex: only run the jobs whose name matches")
    sub.add_parser("status")
    sub.add_parser("reset")
    r = sub.add_parser("reserve")
    r.add_argument("--jobs", required=True)
    r.add_argument("--owner", required=True)
    a = ap.parse_args()
    {"worker": worker, "status": status, "reset": reset, "reserve": reserve}[a.cmd](a)
