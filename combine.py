# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Combine every model's transcripts into one wide CSV: one row per clip, one column per model.

Rows follow the prompts CSV order, with the 7 accents of each prompt in a fixed order. Columns:
seq_no, prompt_no, sentence_id, domain, accent, audio_url, reference, then one column per model
(header = model name) holding that model's raw transcript. A clip a model failed on is left
empty, and its error is listed in <out>.missing.csv.

  uv run combine.py                                                     # open-source models
  uv run combine.py --results results --out results/all_api_models.csv  # API models
  uv run combine.py --results results results_oss --out all_models.csv  # both
--min-ok N keeps only models with at least N transcribed clips.
"""
import argparse
import csv
import glob
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(ROOT, "data", "prompts.csv")
ACCENTS = ["nigerian", "filipino", "indian", "east_african", "mexican", "portuguese", "us"]


def model_names():
    """results_oss folder name -> model name as listed in models.json."""
    path = os.path.join(ROOT, "hf_jobs", "models.json")
    return {m["name"].replace("/", "-"): m["name"] for m in json.load(open(path))} if os.path.exists(path) else {}


def main():
    global DATA_CSV
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", nargs="+", default=[os.path.join(ROOT, "results_oss")],
                    help="one or more results folders; their models are combined side by side")
    ap.add_argument("--out", default=os.path.join(ROOT, "results_oss", "all_open_source_models.csv"))
    ap.add_argument("--min-ok", type=int, default=0, help="only models with at least this many transcribed clips")
    ap.add_argument("--data", default=DATA_CSV, help="prompts CSV")
    args = ap.parse_args()
    DATA_CSV = args.data
    names = model_names()

    models = {}  # display name -> {audio_url: record}
    for d in [d for r in args.results for d in sorted(glob.glob(os.path.join(r, "*/")))]:
        folder = os.path.basename(d.rstrip("/"))
        files = [os.path.join(d, f"{a}.jsonl") for a in ACCENTS if os.path.exists(os.path.join(d, f"{a}.jsonl"))]
        if not files:
            continue  # e.g. raw/
        recs = {}
        for f in files:
            for line in open(f, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prev = recs.get(r["audio_filepath"])
                if prev is None or prev.get("status") != "ok" or r.get("status") in ("ok", "requeued"):
                    recs[r["audio_filepath"]] = r
        if sum(r.get("status") == "ok" for r in recs.values()) < args.min_ok:
            print(f"skipped {folder}: not complete")
            continue
        name = names.get(folder, folder)
        if name in models:
            raise SystemExit(f"model {name} appears in more than one results folder")
        models[name] = recs

    rows, missing, seq = [], [], 0
    with open(DATA_CSV, newline="", encoding="utf-8") as fh:
        for prompt_no, p in enumerate(csv.DictReader(fh), 1):
            for a in ACCENTS:
                url = (p.get(f"{a}_audio_url") or "").strip()
                if not url:
                    continue  # clip not in the set
                seq += 1
                row = {"seq_no": seq, "prompt_no": prompt_no, "sentence_id": p["sentence_id"], "domain": p["domain"],
                       "accent": a, "audio_url": url, "reference": p["text"].strip()}
                for name, recs in models.items():
                    r = recs.get(url)
                    if r and r.get("status") == "ok":
                        row[name] = r.get("pred_text") or ""
                    else:
                        row[name] = ""
                        missing.append({"seq_no": seq, "sentence_id": p["sentence_id"], "accent": a, "model": name,
                                        "status": r.get("status") if r else "missing",
                                        "error": (r or {}).get("error") or ""})
                rows.append(row)

    fields = ["seq_no", "prompt_no", "sentence_id", "domain", "accent", "audio_url", "reference", *models]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:  # BOM so Excel reads UTF-8 correctly
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        w.writerows(rows)
    miss_path = os.path.splitext(args.out)[0] + ".missing.csv"
    with open(miss_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, ["seq_no", "sentence_id", "accent", "model", "status", "error"])
        w.writeheader()
        w.writerows(missing)
    print(f"{args.out}: {len(rows)} clips x {len(models)} models; {len(missing)} missing transcripts ({miss_path})")


if __name__ == "__main__":
    main()
