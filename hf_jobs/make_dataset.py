# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub==2.4.0", "datasets==3.6.0", "soundfile==0.14.0", "librosa==1.0.1"]
# ///
"""Upload the entity-eval audio as a PRIVATE Hugging Face dataset for the open-source model runs.

The open-source models run through the Open ASR Leaderboard's own scripts, which read audio from a
Hub dataset. This builds that dataset from the same 16 kHz mono WAVs the API models received
(audio_cache/, filled by any API script), one config per accent, split "test":

  id           "<sentence_id>_<accent>"; the leaderboard writes it back as `audio_filepath`
  audio        16 kHz mono PCM16 WAV
  text         reference sentence
  sentence_id, domain, accent, audio_url
plus an "all" config that combines the 7 accents (no extra upload).

Safety: the repo is created private, and the upload is refused unless the Hub reports it private.
After the upload every config is read back and checked for row count, unique ids and audio length.

Run: uv run hf_jobs/make_dataset.py --repo <namespace>/<private-dataset>
"""
import argparse
import csv
import hashlib
import io
import os

import soundfile as sf
from datasets import Audio, Dataset, Features, Value, load_dataset
from huggingface_hub import DatasetCard, HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(ROOT, "data", "prompts.csv")
ACCENTS = ["nigerian", "filipino", "indian", "east_african", "mexican", "portuguese", "us"]
SR = 16000
FEATURES = Features({"id": Value("string"), "audio": Audio(sampling_rate=SR), "text": Value("string"),
                     "sentence_id": Value("string"), "domain": Value("string"), "accent": Value("string"),
                     "audio_url": Value("string")})


def load_env(path=os.path.join(ROOT, ".env")):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def rows_for(accent, cache_dir):
    rows, missing = [], []
    with open(DATA_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            url = (r.get(f"{accent}_audio_url") or "").strip()
            if not url:
                continue
            path = os.path.join(cache_dir, hashlib.sha1(url.encode()).hexdigest() + ".wav")
            if not os.path.exists(path):
                missing.append(url)
                continue
            wav = open(path, "rb").read()
            info = sf.info(io.BytesIO(wav))
            if info.samplerate != SR or info.channels != 1:
                raise SystemExit(f"{path}: expected 16 kHz mono, got {info.samplerate} Hz x{info.channels}")
            cid = f"{r['sentence_id']}_{accent}"
            rows.append({"id": cid, "audio": {"bytes": wav, "path": f"{cid}.wav"}, "text": r["text"].strip(),
                         "sentence_id": r["sentence_id"], "domain": r["domain"], "accent": accent,
                         "audio_url": url})
    if missing:
        raise SystemExit(f"{accent}: {len(missing)} clips are not in {cache_dir}; run any API script "
                         f"(e.g. api/deepgram_nova_3.py --limit 1) to fill the cache first")
    return rows


def main():
    global DATA_CSV
    load_env()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True, help="namespace/name of the private dataset repo")
    ap.add_argument("--accents", nargs="+", default=ACCENTS, choices=ACCENTS)
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "audio_cache"))
    ap.add_argument("--data", default=DATA_CSV, help="prompts CSV")
    args = ap.parse_args()
    DATA_CSV = args.data
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set (add it to .env)")
    api = HfApi(token=token)

    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(args.repo, repo_type="dataset").private:
        raise SystemExit(f"{args.repo} exists and is PUBLIC; refusing to upload. Make it private first.")

    expected = {}
    for accent in args.accents:
        rows = rows_for(accent, args.cache_dir)
        expected[accent] = {r["id"]: len(sf.read(io.BytesIO(r["audio"]["bytes"]))[0]) for r in rows}
        ds = Dataset.from_list(rows, features=FEATURES)
        ds.push_to_hub(args.repo, config_name=accent, split="test", private=True, token=token)
        print(f"uploaded {accent}: {len(ds)} clips", flush=True)

    # "all": every accent in one config (no extra upload; it points at the accent files), so a model
    # can be loaded once for the whole set. The accent is part of each id.
    card = DatasetCard.load(args.repo, repo_type="dataset", token=token)
    configs = [c for c in card.data.get("configs", []) if c.get("config_name") != "all"]
    configs.append({"config_name": "all", "data_files": [{"split": "test", "path": [f"{a}/test-*" for a in ACCENTS]}]})
    card.data["configs"] = configs
    card.push_to_hub(args.repo, repo_type="dataset", token=token)

    if not api.repo_info(args.repo, repo_type="dataset").private:
        raise SystemExit(f"{args.repo} is not private!")
    if set(args.accents) == set(ACCENTS):
        ids = load_dataset(args.repo, "all", split="test", token=token)["id"]
        want = {i for w in expected.values() for i in w}
        if len(ids) != len(want) or set(ids) != want:
            raise SystemExit(f"all: read-back mismatch ({len(ids)} rows vs {len(want)} expected)")
        print(f"verified all: {len(ids)} clips")
    for accent, want in expected.items():  # read back from the Hub and compare
        ds = load_dataset(args.repo, accent, split="test", token=token)
        ids = ds["id"]
        if len(ids) != len(want) or set(ids) != set(want) or len(set(ids)) != len(ids):
            raise SystemExit(f"{accent}: read-back mismatch ({len(ids)} rows vs {len(want)} expected)")
        bad = [r["id"] for r in ds if len(r["audio"]["array"]) != want[r["id"]] or r["audio"]["sampling_rate"] != SR]
        if bad:
            raise SystemExit(f"{accent}: {len(bad)} clips differ after upload, e.g. {bad[:3]}")
        print(f"verified {accent}: {len(ids)} clips, ids and audio lengths match")
    print(f"done: https://huggingface.co/datasets/{args.repo} (private)")


if __name__ == "__main__":
    main()
