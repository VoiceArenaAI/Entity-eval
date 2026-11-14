# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub==2.4.0", "soundfile==0.14.0"]
# ///
"""Collect and verify the open-source model results from the private results bucket.

Downloads <run>/ from the bucket and checks every model's manifests against the prompts CSV:
  - each clip id appears exactly once, and (for a full run) every clip in the prompts CSV is present;
  - the reference `text` in the manifest equals the CSV sentence;
  - the manifest `duration` matches the local 16 kHz audio file (within 0.05 s), proving the model got
    the right clip.
Models whose leaderboard script writes no clip id (sample_N) are run per accent and matched by
reference text, which is unique per accent, with the duration as a cross-check.

Writes results_oss/<model_folder>/<accent>.jsonl and .csv in the same layout as the API results,
plus results_oss/summary_<run>.csv. Nothing is uploaded anywhere.

  uv run hf_jobs/collect.py --bucket NS/BUCKET [--run full|smoke-2]
"""
import argparse
import collections
import csv
import hashlib
import json
import os

import soundfile as sf
from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_CSV = os.path.join(ROOT, "data", "prompts.csv")
ACCENTS = ["nigerian", "filipino", "indian", "east_african", "mexican", "portuguese", "us"]
OUT = os.path.join(ROOT, "results_oss")
CSV_FIELDS = ["sentence_id", "domain", "accent", "audio_filepath", "text", "pred_text",
              "duration", "time", "status", "error"]


def load_env(path=os.path.join(ROOT, ".env")):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def expected_clips(cache_dir):
    """id -> clip info from the prompts CSV, with each clip's duration from the audio cache.

    Also returns the clips left out of the set (no URL in the CSV), as ids and (accent, text) keys,
    so their rows in an older run are skipped rather than reported as unmatched."""
    clips, excluded = {}, set()
    with open(DATA_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for a in ACCENTS:
                url = (r.get(f"{a}_audio_url") or "").strip()
                if not url:  # clip not in the set
                    excluded.update({f"{r['sentence_id']}_{a}", (a, r["text"].strip())})
                    continue
                path = os.path.join(cache_dir, hashlib.sha1(url.encode()).hexdigest() + ".wav")
                info = sf.info(path)
                clips[f"{r['sentence_id']}_{a}"] = {"sentence_id": r["sentence_id"], "domain": r["domain"],
                                                    "accent": a, "url": url, "text": r["text"].strip(),
                                                    "duration": info.frames / info.samplerate}
    return clips, excluded


def download(api, bucket, run, dest):
    files = [f for f in api.list_bucket_tree(bucket, prefix=run, recursive=True) if f.path.endswith(".jsonl")]
    api.download_bucket_files(bucket, [(f, os.path.join(dest, f.path)) for f in files], raise_on_missing_files=True)
    return [f.path for f in files]


def main():
    global DATA_CSV
    load_env()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--run", default="full")
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "audio_cache"))
    ap.add_argument("--data", default=DATA_CSV, help="prompts CSV")
    args = ap.parse_args()
    DATA_CSV = args.data
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    if not api.bucket_info(args.bucket).private:
        raise SystemExit(f"bucket {args.bucket} is not private")
    full = args.run == "full"
    clips, excluded = expected_clips(args.cache_dir)
    by_text = {(c["accent"], c["text"]): cid for cid, c in clips.items()}
    raw = os.path.join(OUT, "raw")
    paths = download(api, args.bucket, args.run, raw)

    per_model = collections.defaultdict(list)  # model_folder -> [(config, manifest path)]
    for p in paths:
        _, model, config, _ = p.split("/", 3)
        per_model[model].append((config, os.path.join(raw, p)))

    summary = []
    for model, manifests in sorted(per_model.items()):
        rows, problems = {}, []
        for config, path in manifests:
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                cid = r["audio_filepath"]
                if cid in excluded or (cid not in clips and (config, (r.get("text") or "").strip()) in excluded):
                    continue
                if cid not in clips:  # no id written: match by reference text within the accent
                    cid = by_text.get((config, (r.get("text") or "").strip()))
                if cid is None:
                    problems.append(f"{config}: unmatched row {r.get('audio_filepath')}")
                    continue
                c = clips[cid]
                if config != "all" and c["accent"] != config:
                    problems.append(f"{cid} found in config {config}")
                if (r.get("text") or "").strip() != c["text"]:
                    problems.append(f"{cid}: reference text differs from the CSV")
                if abs(float(r["duration"]) - c["duration"]) > 0.05:
                    problems.append(f"{cid}: duration {r['duration']} != {c['duration']:.3f}")
                if cid in rows:
                    problems.append(f"{cid}: duplicated")
                rows[cid] = r
        for a in ACCENTS:
            want = [cid for cid, c in clips.items() if c["accent"] == a]
            got = [cid for cid in want if cid in rows]
            if full and len(got) != len(want):
                problems.append(f"{a}: {len(got)}/{len(want)} clips")
            if not got:
                continue
            os.makedirs(os.path.join(OUT, model), exist_ok=True)
            out = []
            for cid in got:
                c, r = clips[cid], rows[cid]
                out.append({"sentence_id": c["sentence_id"], "domain": c["domain"], "accent": a,
                            "audio_filepath": c["url"], "text": c["text"], "pred_text": r.get("pred_text") or "",
                            "duration": round(c["duration"], 3), "time": r.get("time"), "status": "ok",
                            "error": None, "model": model, "clip_id": cid})
            with open(os.path.join(OUT, model, f"{a}.jsonl"), "w", encoding="utf-8") as fh:
                fh.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in out)
            with open(os.path.join(OUT, model, f"{a}.csv"), "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, CSV_FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(out)
        empty = sum(1 for r in rows.values() if not (r.get("pred_text") or "").strip())
        summary.append({"model": model, "clips": len(rows), "empty_pred": empty, "problems": len(problems),
                        "first_problems": " | ".join(problems[:3])})
        flag = "OK " if not problems else "BAD"
        print(f"{flag} {model:55s} clips={len(rows):5d} empty={empty:3d} problems={len(problems)} {problems[:2]}")

    with open(os.path.join(OUT, f"summary_{args.run}.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["model", "clips", "empty_pred", "problems", "first_problems"])
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    main()
