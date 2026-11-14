# /// script
# requires-python = ">=3.12"
# dependencies = ["aiohttp==3.14.5", "websockets==17.4", "numpy==2.5.4", "soundfile==0.14.0", "soxr==1.1.0",
#                 "msgpack==1.2.4", "tqdm==4.70.1"]
# ///
"""Amazon Transcribe (batch), direct REST API with SigV4 (no SDK).

Endpoint: PUT https://<bucket>.s3.<region>.amazonaws.com/entity-eval/<job>.wav, then
          POST https://transcribe.<region>.amazonaws.com/ (StartTranscriptionJob, GetTranscriptionJob)
Docs:     https://docs.aws.amazon.com/transcribe/latest/APIReference/API_StartTranscriptionJob.html
Flow:     upload the 16 kHz WAV to S3 -> start job -> poll every 5 s -> read the transcript from the
          presigned TranscriptFileUri -> delete the uploaded WAV.
Request:  LanguageCode=en-US (Transcribe requires a locale), Media.MediaFileUri.
          All other parameters are API defaults.
Needs:    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, AWS_S3_BUCKET in .env; IAM s3:PutObject and
          s3:DeleteObject on the bucket, transcribe:StartTranscriptionJob and transcribe:GetTranscriptionJob.

Standalone: imports nothing from this repo; dependencies are in the header above, keys in .env.
Run:      uv run api/aws_transcribe.py [--limit N] [--accents us indian ...]
"""
import argparse
import asyncio
import contextlib
import csv
import datetime
import fcntl
import hashlib
import hmac
import io
import json
import os
import random
import signal
import time
import uuid
from dataclasses import dataclass
from urllib.parse import urlsplit

import aiohttp
import numpy as np
import soundfile as sf
import soxr
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(ROOT, "data", "prompts.csv")
ACCENTS = ["nigerian", "filipino", "indian", "east_african", "mexican", "portuguese", "us"]
SR = 16000


REGION = os.environ.get("AWS_REGION", "us-east-1")
PREFIX = "entity-eval"  # S3 key prefix and job-name prefix
POLL_S = 5
JOB_TIMEOUT_S = 1800
START_ACTION, GET_ACTION = "StartTranscriptionJob", "GetTranscriptionJob"
NAME_FIELD, JOB_FIELD, STATUS_FIELD = "TranscriptionJobName", "TranscriptionJob", "TranscriptionJobStatus"
# Transcribe reports most errors as HTTP 400 with an exception name in `__type`.
ERRORS = {"LimitExceededException": 429, "ThrottlingException": 429, "InternalFailureException": 500,
          "ServiceUnavailableException": 503, "ConflictException": 409, "AccessDeniedException": 403,
          "UnrecognizedClientException": 401, "InvalidSignatureException": 403, "ExpiredTokenException": 401}


def sign(method, url, service, creds, body=b"", headers=None):
    """SigV4 headers for one request (the URL path must already be URI-encoded; no query string)."""
    region, access_key, secret_key, token = creds
    u = urlsplit(url)
    now = datetime.datetime.now(datetime.timezone.utc)
    amz_date, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    h = {"host": u.netloc, "x-amz-date": amz_date, "x-amz-content-sha256": hashlib.sha256(body).hexdigest(),
         **{k.lower(): v for k, v in (headers or {}).items()}}
    if token:
        h["x-amz-security-token"] = token
    signed = ";".join(sorted(h))
    canonical = "\n".join([method, u.path or "/", "", "".join(f"{k}:{h[k]}\n" for k in sorted(h)), signed,
                           h["x-amz-content-sha256"]])
    scope = f"{day}/{region}/{service}/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    key = ("AWS4" + secret_key).encode()
    for part in (day, region, service, "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    h["authorization"] = (f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, SignedHeaders={signed}, "
                          f"Signature={hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()}")
    del h["host"]  # aiohttp sends the same value
    return h


class Model:
    def __init__(self):
        self.creds = (REGION, env("AWS_ACCESS_KEY_ID"), env("AWS_SECRET_ACCESS_KEY"), os.environ.get("AWS_SESSION_TOKEN"))
        self.bucket = env("AWS_S3_BUCKET")
        self.s3_url = f"https://{self.bucket}.s3.{REGION}.amazonaws.com"
        self.api_url = f"https://transcribe.{REGION}.amazonaws.com/"

    async def s3(self, http, method, key, body=b""):
        url = f"{self.s3_url}/{key}"
        async with http.request(method, url, data=body or None, headers=sign(method, url, "s3", self.creds, body)) as r:
            await raise_for_status(r)
            return await r.read()

    async def call(self, http, action, payload):
        body = json.dumps(payload).encode()
        headers = sign("POST", self.api_url, "transcribe", self.creds, body,
                       {"content-type": "application/x-amz-json-1.1", "x-amz-target": f"Transcribe.{action}"})
        async with http.post(self.api_url, data=body, headers=headers) as r:
            text = await r.text()
            if r.status >= 300:
                try:
                    kind = json.loads(text).get("__type", "").split("#")[-1]
                except ValueError:
                    kind = ""
                raise ApiError(ERRORS.get(kind, r.status), f"{action}: {text}", retry_after(r.headers))
            return json.loads(text)

    async def wait(self, http, name):
        deadline = time.monotonic() + JOB_TIMEOUT_S
        while time.monotonic() < deadline:
            job = (await self.call(http, GET_ACTION, {NAME_FIELD: name}))[JOB_FIELD]
            status = job[STATUS_FIELD]
            if status == "COMPLETED":
                return job
            if status == "FAILED":
                raise PermanentError(f"job {name} failed: {job.get('FailureReason')}")
            await asyncio.sleep(POLL_S)
        raise ApiError(504, f"job {name} did not finish in {JOB_TIMEOUT_S} s")

    async def transcribe(self, clip, http):
        name = f"{PREFIX}-{clip.sentence_id}-{clip.accent}-{uuid.uuid4().hex[:8]}"
        key = f"{PREFIX}/{name}.wav"
        await self.s3(http, "PUT", key, clip.wav)
        try:
            await self.call(http, START_ACTION, {NAME_FIELD: name, "LanguageCode": "en-US",
                                                 "Media": {"MediaFileUri": f"s3://{self.bucket}/{key}"}})
            job = await self.wait(http, name)
            # No OutputBucketName: Transcribe keeps the transcript and returns a presigned URL to it.
            async with http.get(job["Transcript"]["TranscriptFileUri"]) as r:
                await raise_for_status(r)
                out = await r.json(content_type=None)
            return {"text": out["results"]["transcripts"][0]["transcript"], "job_name": name}
        finally:
            with contextlib.suppress(Exception):
                await self.s3(http, "DELETE", key)


# ------------------------ shared runner: identical in every script ------------------------
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
ABORT_STATUS = {401, 402, 403}
MAX_CONSECUTIVE_FAILURES = 30


class ApiError(Exception):
    """HTTP-style failure; `status` decides between retry, skip and abort."""

    def __init__(self, status, message, retry_after=None):
        super().__init__(f"HTTP {status}: {str(message)[:500]}")
        self.status, self.retry_after = status, retry_after


class PermanentError(Exception):
    """Deterministic failure for this clip: record it, do not retry."""


class AbortRun(Exception):
    """Account-level failure (auth, billing): stop the whole run."""


def load_env(path=os.path.join(ROOT, ".env")):
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def env(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set (add it to .env)")
    return value


def retry_after(headers):
    try:
        return max(0.0, float(headers.get("Retry-After")))
    except (TypeError, ValueError):
        return None


async def raise_for_status(resp):
    """Raise ApiError for a non-2xx aiohttp response."""
    if resp.status >= 300:
        body = await resp.text()
        # S3-style presigned uploads report a slow connection as 400 RequestTimeout: transient.
        status = 408 if resp.status == 400 and "<Code>RequestTimeout</Code>" in body else resp.status
        raise ApiError(status, body, retry_after(resp.headers))


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


def to_wav16k(raw):
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SR:
        mono = soxr.resample(mono, sr, SR)
    buf = io.BytesIO()
    sf.write(buf, np.clip(mono, -1.0, 1.0), SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class AudioCache:
    """Downloads each URL once, stores it as 16 kHz WAV; safe to share between processes."""

    def __init__(self, http, cache_dir):
        self.http, self.dir = http, cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def path(self, url):
        return os.path.join(self.dir, hashlib.sha1(url.encode()).hexdigest() + ".wav")

    async def load(self, clip, retries=5):
        path = self.path(clip.url)
        if not os.path.exists(path):
            for attempt in range(retries + 1):
                try:
                    async with self.http.get(clip.url, timeout=aiohttp.ClientTimeout(total=120)) as r:
                        if r.status in (403, 404):
                            raise PermanentError(f"audio download HTTP {r.status}")
                        r.raise_for_status()
                        raw = await r.read()
                    break
                except PermanentError:
                    raise
                except Exception:
                    if attempt == retries:
                        raise
                    await asyncio.sleep(2 ** attempt + random.random())
            wav = await asyncio.get_running_loop().run_in_executor(None, to_wav16k, raw)
            tmp = f"{path}.{os.getpid()}.{random.randrange(1 << 30)}.tmp"
            with open(tmp, "wb") as fh:
                fh.write(wav)
            os.replace(tmp, path)  # atomic: concurrent runners never see a partial file
        with open(path, "rb") as fh:
            clip.wav = fh.read()
        clip.pcm = sf.read(io.BytesIO(clip.wav), dtype="int16")[0].tobytes()


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


CSV_FIELDS = ["sentence_id", "domain", "accent", "audio_filepath", "text", "pred_text",
              "duration", "time", "status", "error"]


def atomic_write(path, write):
    """Write via a temp file + rename, so a crash never leaves a half-written file."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        write(fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_outputs(out_dir, accent, clips):
    """Rewrite <accent>.jsonl (one record per clip, input order) and <accent>.csv.

    Never drops a record: rows for clips outside `clips` are kept at the end.
    """
    path = os.path.join(out_dir, f"{accent}.jsonl")
    recs = read_jsonl(path)
    urls = [c.url for c in clips]
    known = set(urls)
    rows = [recs[u] for u in urls if u in recs] + [r for u, r in recs.items() if u not in known]
    atomic_write(path, lambda fh: fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    def write_csv(fh):
        w = csv.DictWriter(fh, CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    atomic_write(os.path.join(out_dir, f"{accent}.csv"), write_csv)
    ok = sum(r["status"] == "ok" for r in rows)
    return {"accent": accent, "ok": ok, "error": len(rows) - ok, "pending": sum(u not in recs for u in urls)}


class Runner:
    def __init__(self, provider, name, args):
        self.p, self.name, self.args = provider, name, args
        self.out_dir = os.path.join(args.out_dir, name)
        self.cooldown_until = 0.0
        self.consecutive_failures = 0
        self.stop = None  # reason the run stopped early

    async def transcribe(self, http, clip):
        """Call the provider with retries; return (result, attempts, errors of the failed attempts)."""
        errors = []
        for attempt in range(1, self.args.max_retries + 2):
            while (wait := self.cooldown_until - time.monotonic()) > 0:
                await asyncio.sleep(wait)
            try:
                return await self.p.transcribe(clip, http), attempt, errors
            except (PermanentError, AbortRun):
                raise
            except ApiError as e:
                if e.status in ABORT_STATUS:
                    raise AbortRun(str(e)) from e
                if e.status not in RETRY_STATUS:
                    raise PermanentError(str(e)) from e
                delay = e.retry_after if e.retry_after is not None else 2 ** attempt
                if e.status == 429:  # rate limited: back the whole pool off
                    self.cooldown_until = max(self.cooldown_until, time.monotonic() + min(delay, 30))
                errors.append(str(e)[:300])
            except Exception as e:  # network errors, timeouts, dropped sockets: retry
                delay = 2 ** attempt
                errors.append(f"{type(e).__name__}: {e}"[:300])
            if attempt > self.args.max_retries:
                raise RuntimeError(f"failed after {attempt} attempts; last: {errors[-1]}")
            await asyncio.sleep(min(delay, 60) + random.random())

    async def process(self, http, cache, clip, files, bar):
        t_start = time.perf_counter()
        rec = {"audio_filepath": clip.url, "duration": None, "time": None, "text": clip.text,
               "pred_text": "", "sentence_id": clip.sentence_id, "domain": clip.domain,
               "accent": clip.accent, "model": self.name}
        try:
            await cache.load(clip)
            rec["duration"] = round(clip.duration, 3)
            t0 = time.perf_counter()
            result, attempts, errors = await self.transcribe(http, clip)
            text = result.pop("text")
            rec.update(pred_text=(text or "").strip(), time=round(time.perf_counter() - t0, 3),
                       status="ok", error=None, attempts=attempts, retry_errors=errors or None,
                       response=result)
            self.consecutive_failures = 0
        except AbortRun as e:
            self.stop = f"aborted (account-level error): {e}"
            return None
        except Exception as e:
            rec.update(time=round(time.perf_counter() - t_start, 3), status="error",
                       error=f"{type(e).__name__}: {e}"[:800])
            self.consecutive_failures += 1
            if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                self.stop = f"stopped after {MAX_CONSECUTIVE_FAILURES} consecutive failures; last: {rec['error']}"
        fh = files[clip.accent]
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())  # the record survives a crash or power loss
        bar.update(1)
        if rec["status"] != "ok":
            tqdm.write(f"[{self.name}] {clip.accent}/{clip.sentence_id}: {rec['error'][:300]}")
        clip.wav = clip.pcm = b""  # keep memory flat
        return rec

    async def run(self):
        a = self.args
        clips = load_clips(a.data, a.accents, a.limit)
        os.makedirs(self.out_dir, exist_ok=True)
        lock = open(os.path.join(self.out_dir, ".lock"), "w")
        try:  # one process per results folder: two writers would race on the same files
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"[{self.name}] another process is already writing {self.out_dir}") from None
        done = {}
        for accent in a.accents:
            for url, r in read_jsonl(os.path.join(self.out_dir, f"{accent}.jsonl")).items():
                rejected = (r["error"] or "").startswith("PermanentError")
                if r["status"] == "ok" or (rejected and not a.retry_failed):
                    done[url] = r
        todo = [c for c in clips if c.url not in done]
        print(f"[{self.name}] clips={len(clips)} done={len(clips) - len(todo)} todo={len(todo)} "
              f"concurrency={a.concurrency}", flush=True)

        loop = asyncio.get_running_loop()
        draining = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, draining.set)  # finish in-flight clips, start no new ones

        files = {acc: open(os.path.join(self.out_dir, f"{acc}.jsonl"), "a", encoding="utf-8")
                 for acc in a.accents}
        bar = tqdm(total=len(todo), desc=self.name, mininterval=5, dynamic_ncols=True)
        timeout = aiohttp.ClientTimeout(total=a.request_timeout, sock_connect=30)
        try:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), timeout=timeout) as http:
                cache = AudioCache(http, a.cache_dir)
                queue = list(reversed(todo))
                if queue:  # one clip first: a bad key, model id or parameter fails before fanning out
                    first = await self.process(http, cache, queue.pop(), files, bar)
                    if first and first["status"] != "ok" and self.stop is None:
                        self.stop = f"first clip failed, not fanning out: {first['error']}"

                async def worker():
                    while queue and self.stop is None and not draining.is_set():
                        await self.process(http, cache, queue.pop(), files, bar)

                await asyncio.gather(*(worker() for _ in range(a.concurrency)))
        finally:
            bar.close()
            for fh in files.values():
                fh.close()
        all_clips = load_clips(a.data, a.accents)  # full set, so a --limit run keeps earlier rows
        print(f"\n{'accent':<14}{'ok':>6}{'error':>7}{'pending':>9}")
        for acc in a.accents:
            n = write_outputs(self.out_dir, acc, [c for c in all_clips if c.accent == acc])
            print(f"{acc:<14}{n['ok']:>6}{n['error']:>7}{n['pending']:>9}")
        print(f"results: {self.out_dir}")
        if self.stop:
            raise SystemExit(f"[{self.name}] {self.stop}")


def main(provider_factory, name, concurrency):
    """Entry point for every model script.

    provider_factory(args) builds the provider after .env is loaded, so a missing key fails fast.
    """
    load_env()
    ap = argparse.ArgumentParser(description=f"Transcribe the entity eval set with {name}")
    ap.add_argument("--accents", nargs="+", default=ACCENTS, choices=ACCENTS)
    ap.add_argument("--limit", type=int, help="only the first N sentences (smoke test)")
    ap.add_argument("--concurrency", type=int, default=concurrency)
    ap.add_argument("--speed", type=float, default=1.0, help="streaming only: 1.0 = real time")
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--retry-failed", action="store_true", help="also resend clips rejected with a 4xx")
    ap.add_argument("--request-timeout", type=float, default=300)
    ap.add_argument("--data", default=DATA_CSV)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "audio_cache"))
    args = ap.parse_args()
    asyncio.run(Runner(provider_factory(args), name, args).run())


# ------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main(lambda args: Model(), "aws_transcribe", concurrency=40)
