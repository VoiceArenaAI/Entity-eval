# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub==2.4.0"]
# ///
"""Run open-source ASR models on Hugging Face Jobs with the Open ASR Leaderboard's own setup.

Each model in models.json runs in its leaderboard Space image, with the leaderboard repo's eval
script and normalizer/ injected into the job exactly as the leaderboard's submit scripts do by
default (USE_LOCAL_SCRIPT=1, USE_LOCAL_NORMALIZER=1), and with the same arguments. Only the dataset
changes. Models the leaderboard repo does not cover use a runner from runners/ ("inject_local"),
which loads data and writes manifests through the same normalizer/ code. The dataset and the results bucket must both be PRIVATE, and the script refuses to run
otherwise.

Results land in the bucket under <run>/<model_folder>/<config>/. A model runs either one job on the
dataset's "all" config, or one job per accent config ("mode" in models.json). Relaunching is safe:
a (model, config) that already has a manifest in the bucket, or a job still running, is skipped.

  uv run hf_jobs/run_jobs.py launch --dataset NS/DATA --bucket NS/BUCKET --leaderboard PATH [--models ...] [--limit N]
  uv run hf_jobs/run_jobs.py status --bucket NS/BUCKET [--run RUN]

--limit N runs N clips per job (smoke test) into run "smoke-N" instead of "full"; --configs picks
the configs (e.g. --configs us for a quick test).
"""
import argparse
import base64
import io
import json
import os
import re
import subprocess
import tarfile
import time

from huggingface_hub import HfApi, Volume

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ACCENTS = ["nigerian", "filipino", "indian", "east_african", "mexican", "portuguese", "us"]
STATE = os.path.join(ROOT, "results_oss", "jobs.jsonl")
FINISHED = {"COMPLETED", "ERROR", "CANCELED", "CANCELLED", "DELETED"}  # any other stage counts as active
LEADERBOARD_COMMIT = "859159c"  # open_asr_leaderboard commit the settings in models.json were taken from


def load_env(path=os.path.join(ROOT, ".env")):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def normalizer_b64(leaderboard):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(os.path.join(leaderboard, "normalizer"), arcname="normalizer",
                filter=lambda t: None if "__pycache__" in t.name or t.name.endswith(".pyc") else t)
    return base64.b64encode(buf.getvalue()).decode()


def folder_of(name):
    return name.replace("/", "-")


def label_of(key):
    """Job labels allow only [a-zA-Z0-9_-]."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", key)


def job_script(m, leaderboard, dataset, config, out_dir, limit):
    """The bash the job runs: inject files, run the leaderboard command, copy the manifest out."""
    flags = f" {m['normalizer_tar_flags']}" if m.get("normalizer_tar_flags") else ""
    steps = ["mkdir -p /app && cd /app",  # Space images already work in /app; plain images need it created
             f"echo '{normalizer_b64(leaderboard)}' | base64 -d | tar{flags} -xzf - -C /app"]
    for f in m["inject"]:
        src = open(os.path.join(leaderboard, m["folder"], f), encoding="utf-8").read()
        for old, new in m.get("patch", {}).get(f, []):  # documented fixes, see "patch" in models.json
            if src.count(old) != 1:
                raise SystemExit(f"{m['name']}: patch for {f} does not apply cleanly")
            src = src.replace(old, new)
        steps.append(f"echo '{base64.b64encode(src.encode()).decode()}' | base64 -d > /app/{f}")
    for f in m.get("inject_local", []):  # runners for models the leaderboard repo does not cover
        src = open(os.path.join(HERE, f), encoding="utf-8").read()
        steps.append(f"echo '{base64.b64encode(src.encode()).decode()}' | base64 -d > /app/{os.path.basename(f)}")
    steps += m.get("pre", [])
    args = [a.replace("{DATASET_PATH}", dataset).replace("{DATASET}", config).replace("{SPLIT}", "test")
            for a in m["args"]]
    if limit:
        args = [a for a in args if not a.startswith("--max_eval_samples")] + [f"--max_eval_samples={limit}"]
    steps.append(f"mkdir -p {out_dir}")
    if m["kind"] == "transcribe":  # run_transcribe.sh starts vLLM, evaluates, and copies the manifest itself
        steps.append(f"RESULTS_DEST_DIR={out_dir} {m['command']} {' '.join(args)}")
    else:
        steps.append(f"PYTHONPATH=/app {m['command']} {' '.join(args)}")
        steps.append(f"cp results/*.jsonl {out_dir}/")
    steps.append(f"ls -la {out_dir}")
    return "set -e\n" + "\n".join(steps)


def require_private(api, dataset, bucket):
    if not api.repo_info(dataset, repo_type="dataset").private:
        raise SystemExit(f"dataset {dataset} is not private; refusing to run")
    if not api.bucket_info(bucket).private:
        raise SystemExit(f"bucket {bucket} is not private; refusing to run")


def launch(api, args, models, token):
    require_private(api, args.dataset, args.bucket)
    head = subprocess.run(["git", "-C", args.leaderboard, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if not head.startswith(LEADERBOARD_COMMIT) and not args.any_commit:
        raise SystemExit(f"leaderboard clone is at {head}, models.json was taken from {LEADERBOARD_COMMIT} "
                         f"(git checkout {LEADERBOARD_COMMIT}, or pass --any-commit)")
    run = f"smoke-{args.limit}" if args.limit else "full"
    have = {f.path.rsplit("/", 1)[0] for f in api.list_bucket_tree(args.bucket, prefix=run, recursive=True)
            if f.path.endswith(".jsonl")}
    active = {(j.labels or {}).get("key") for j in api.list_jobs(labels={"project": "entity-eval", "run": label_of(run)})
              if j.status.stage not in FINISHED}
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    for m in models:
        configs = args.configs or (ACCENTS if m["mode"] == "per_accent" else ["all"])
        for config in configs:
            key = f"{run}/{folder_of(m['name'])}/{config}"
            if key in have or label_of(key) in active:
                print(f"skip {key} ({'done' if key in have else 'running'})")
                continue
            job = api.run_job(
                image=m.get("image") or f"hf.co/spaces/{m['space']}",
                command=["bash", "-c", job_script(m, args.leaderboard, args.dataset, config, f"/results/{key}", args.limit)],
                env=m.get("env") or None, secrets={"HF_TOKEN": token}, flavor=m["flavor"], timeout="8h",
                labels={"project": "entity-eval", "run": label_of(run), "key": label_of(key)},
                volumes=[Volume(type="bucket", source=args.bucket, mount_path="/results")])
            rec = {"key": key, "model": m["name"], "config": config, "job_id": job.id, "url": job.url,
                   "submitted": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(STATE, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"launched {key}: {job.url}")


def status(api, args):
    run = args.run
    have = {label_of(f.path.rsplit("/", 1)[0]) for f in api.list_bucket_tree(args.bucket, prefix=run, recursive=True)
            if f.path.endswith(".jsonl")}
    jobs = {}
    for j in api.list_jobs(labels={"project": "entity-eval", "run": label_of(run)}):
        key = (j.labels or {}).get("key")
        rank = (j.status.stage != "CANCELED", j.created_at)  # latest attempt that was not cancelled
        if key and (key not in jobs or rank > (jobs[key].status.stage != "CANCELED", jobs[key].created_at)):
            jobs[key] = j
    counts = {}
    for key, j in sorted(jobs.items()):
        stage = "DONE+RESULT" if key in have else j.status.stage
        counts[stage] = counts.get(stage, 0) + 1
        if stage not in ("DONE+RESULT", "RUNNING"):
            print(f"{stage:12s} {key}  {j.status.message or ''}  {j.url}")
    print(run, counts)


def main():
    load_env()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    la = sub.add_parser("launch")
    la.add_argument("--dataset", required=True)
    la.add_argument("--bucket", required=True)
    la.add_argument("--leaderboard", required=True, help="path to a clone of huggingface/open_asr_leaderboard")
    la.add_argument("--models", nargs="+", help="subset of model names (default: all in models.json)")
    la.add_argument("--limit", type=int, help="clips per job, for a smoke test")
    la.add_argument("--configs", nargs="+", choices=ACCENTS + ["all"], help="override the configs to run")
    la.add_argument("--any-commit", action="store_true")
    st = sub.add_parser("status")
    st.add_argument("--bucket", required=True)
    st.add_argument("--run", default="full")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set (add it to .env)")
    api = HfApi(token=token)
    if args.cmd == "status":
        return status(api, args)
    models = json.load(open(os.path.join(HERE, "models.json")))
    if args.models:
        unknown = set(args.models) - {m["name"] for m in models}
        if unknown:
            raise SystemExit(f"unknown models: {sorted(unknown)}")
        models = [m for m in models if m["name"] in args.models]
    launch(api, args, models, token)


if __name__ == "__main__":
    main()
