# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Sanity-check transcripts and optionally send suspicious clips again.

A call can succeed and still return an unusable transcript. This flags, per model and accent:
  error        the clip failed every retry (rerunning the model script retries it)
  pending      the clip was never transcribed
  empty        status ok but empty text
  short        fewer than half as many words as the reference (likely truncated)
  long         more than twice as many words as the reference (repetition loop, or spelled-out
               emails / codes, which legitimately double the word count)
  provider     a session that ended without a final result (Gemini Live interim fallback,
               ElevenLabs Live final-commit timeout, Flux turn without EndOfTurn). The current
               scripts retry these instead of recording them; this check is a safety net.

empty / short / long are reported for review only: they are usually what the model really said,
and re-sending them until they look better would bias the eval. Only `provider` rows are requeued.

    uv run check_results.py                     # report; flagged rows -> results/qa_flagged.csv
    uv run check_results.py --requeue           # also mark `provider` rows to be sent again
    uv run api/<model>.py                       # ...then rerun the model: it resends only those
    uv run api/<model>.py --retry-failed        # also resends clips rejected with a 4xx

--requeue appends a "requeued" record (the old record stays in the JSONL history) and requeues a
clip at most once. Models that are currently running are skipped.
"""
import argparse
import csv
import fcntl
import json
import os
from dataclasses import dataclass


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(ROOT, "data", "prompts.csv")
ACCENTS = ["nigerian", "filipino", "indian", "east_african", "mexican", "portuguese", "us"]
SR = 16000


def words(text):
    return len((text or "").split())


def flags(rec):
    if rec["status"] != "ok":
        return [rec["status"]]
    resp, out = rec.get("response") or {}, []
    n, ref = words(rec["pred_text"]), words(rec["text"])
    if n == 0:
        out.append("empty")
    elif n < 0.5 * ref:
        out.append("short")
    elif n > 2 * ref:
        out.append("long")
    if resp.get("used_interim_fallback") or resp.get("final_commit_timeout") or resp.get("unfinalized_turns"):
        out.append("provider")
    return out


def requeued_before(path):
    with open(path, encoding="utf-8") as fh:
        return {json.loads(line)["audio_filepath"] for line in fh if '"requeued"' in line}


def main():
    global DATA_CSV
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--requeue", action="store_true")
    ap.add_argument("--data", default=DATA_CSV, help="prompts CSV")
    args = ap.parse_args()
    DATA_CSV = args.data
    clips = load_clips(DATA_CSV, ACCENTS)
    flagged = []
    print(f"{'model':<44}{'ok':>6}{'error':>7}{'pending':>8}{'empty':>7}{'short':>7}{'long':>6}{'prov':>6}")
    for model in sorted(os.listdir(args.results)):
        mdir = os.path.join(args.results, model)
        if not os.path.isdir(mdir):
            continue
        lock = open(os.path.join(mdir, ".lock"), "a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # held for the rest of this model's check
            running = False
        except BlockingIOError:
            running = True
        counts = dict.fromkeys(["ok", "error", "pending", "empty", "short", "long", "provider"], 0)
        for accent in ACCENTS:
            path = os.path.join(mdir, f"{accent}.jsonl")
            recs = read_jsonl(path)
            done_once = requeued_before(path) if os.path.exists(path) else set()
            requeue = []
            for c in clips:
                if c.accent != accent:
                    continue
                rec = recs.get(c.url)
                f = flags(rec) if rec else ["pending"]
                counts["ok"] += bool(rec and rec["status"] == "ok")
                for k in f:
                    counts[k if k in counts else "error"] += 1
                if f and f != ["pending"]:
                    flagged.append({"model": model, "accent": accent, "sentence_id": c.sentence_id,
                                    "flags": " ".join(f), "audio_filepath": c.url, "text": c.text,
                                    "pred_text": (rec or {}).get("pred_text", "")})
                if rec and rec["status"] == "ok" and "provider" in f and c.url not in done_once:
                    requeue.append({**rec, "status": "requeued", "error": f"QA: {' '.join(f)}",
                                    "previous_pred_text": rec["pred_text"], "pred_text": ""})
            if args.requeue and requeue and not running:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in requeue)
                    fh.flush()
                    os.fsync(fh.fileno())
        lock.close()
        note = "  (running: not requeued)" if running and args.requeue else ""
        print(f"{model:<44}{counts['ok']:>6}{counts['error']:>7}{counts['pending']:>8}{counts['empty']:>7}"
              f"{counts['short']:>7}{counts['long']:>6}{counts['provider']:>6}{note}")
    out = os.path.join(args.results, "qa_flagged.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["model", "accent", "sentence_id", "flags", "audio_filepath", "text", "pred_text"])
        w.writeheader()
        w.writerows(flagged)
    print(f"\n{len(flagged)} flagged rows -> {out}")


# ------------------------ shared runner: identical in every script ------------------------
@dataclass
class Clip:
    sentence_id: str
    domain: str
    accent: str
    url: str
    text: str
    wav: bytes = b""  # 16 kHz mono PCM16 WAV, as sent to batch APIs
    pcm: bytes = b""  # the same samples without the header, as sent to streaming APIs

    @property
    def duration(self):
        return len(self.pcm) / (2 * SR)


def load_clips(data_csv, accents, limit=None):
    """One clip per (sentence, accent); `limit` keeps only the first N sentences."""
    clips = []
    with open(data_csv, newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if limit is not None and i >= limit:
                break
            for accent in accents:
                url = (row.get(f"{accent}_audio_url") or "").strip()
                if url:
                    clips.append(Clip(row["sentence_id"], row["domain"], accent, url, row["text"].strip()))
    return clips


def read_jsonl(path):
    """Latest record per clip.

    A success is never replaced by a later failure, only by a later success or by a
    "requeued" marker written by check_results.py --requeue (which sends the clip again).
    """
    recs = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line torn by a crash
                prev = recs.get(r["audio_filepath"])
                if prev is None or prev["status"] != "ok" or r["status"] in ("ok", "requeued"):
                    recs[r["audio_filepath"]] = r
    return recs


# ------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
