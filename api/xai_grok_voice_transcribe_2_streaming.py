# /// script
# requires-python = ">=3.12"
# dependencies = ["aiohttp==3.14.5", "websockets==17.4", "numpy==2.5.4", "soundfile==0.14.0", "soxr==1.1.0",
#                 "msgpack==1.2.4", "tqdm==4.70.1"]
# ///
"""xAI Grok Voice Transcribe 2.0 (streaming), direct WebSocket API.

Endpoint: wss://api.x.ai/v1/stt?model=grok-voice-transcribe-2.0&sample_rate=16000&encoding=pcm&language=en
Docs:     https://docs.x.ai/developers/model-capabilities/audio/speech-to-text#streaming-speech-to-text-websocket
Protocol: <- transcript.created (ready) ; -> raw PCM16 binary frames paced at real time ; -> {"type": "audio.done"}
          <- transcript.partial* ; <- transcript.done (server then closes). Transcript = the utterance finals
          (is_final+speech_final) in order, plus the text still pending at audio.done (transcript.done text,
          or the chunk finals after the last utterance final).
Request:  model, language=en, audio format (encoding=pcm, sample_rate=16000). All other parameters are API
          defaults (format=false, interim_results=false, endpointing=400 ms, smart_turn off).

Standalone: imports nothing from this repo; dependencies are in the header above, keys in .env.
Run:      uv run api/xai_grok_voice_transcribe_2_streaming.py [--limit N] [--accents us indian ...]
"""
import argparse
import asyncio
import csv
import fcntl
import hashlib
import io
import json
import os
import random
import signal
import time
from dataclasses import dataclass
from urllib.parse import urlencode

import aiohttp
import numpy as np
import soundfile as sf
import soxr
import websockets
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(ROOT, "data", "prompts.csv")
ACCENTS = ["nigerian", "filipino", "indian", "east_african", "mexican", "portuguese", "us"]
SR = 16000
LANGUAGE = "en"


ENDPOINT = "wss://api.x.ai/v1/stt"
PARAMS = {"model": "grok-voice-transcribe-2.0", "sample_rate": SR, "encoding": "pcm", "language": LANGUAGE}


class Model:
    def __init__(self, speed):
        self.speed = speed
        self.headers = {"Authorization": f"Bearer {env('XAI_API_KEY')}"}

    async def session(self, clip):
        async with websockets.connect(f"{ENDPOINT}?{urlencode(PARAMS)}", additional_headers=self.headers,
                                      max_size=None) as ws:
            ready = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if ready.get("type") != "transcript.created":
                raise inband_error(ready, 400)

            async def sender():
                await send_paced(ws.send, clip.pcm, 100, self.speed)
                await ws.send(json.dumps({"type": "audio.done"}))

            async def receiver():
                # is_final + speech_final: the whole utterance, stitched; is_final alone: a ~3 s chunk of the
                # utterance in progress. transcript.done carries only text still pending at audio.done.
                utterances, pending = [], []
                async for msg in ws:
                    m = json.loads(msg)
                    if m.get("type") == "error":
                        raise inband_error(m)
                    if m.get("type") == "transcript.partial" and m.get("is_final"):
                        text = (m.get("text") or "").strip()
                        if m.get("speech_final"):
                            utterances.append(text)
                            pending = []
                        elif text:
                            pending.append(text)
                    elif m.get("type") == "transcript.done":
                        tail = (m.get("text") or "").strip() or " ".join(pending)
                        text = " ".join(t for t in utterances + [tail] if t)
                        return {"text": text, "utterances": len(utterances), "audio_duration_s": m.get("duration")}
                raise ApiError(503, "socket closed before transcript.done")

            return await run_session(sender(), receiver())

    async def transcribe(self, clip, http):
        return await streaming(self.session(clip), clip, self.speed)


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


def inband_error(payload, default=503):
    """ApiError for an error message received inside a streaming session, classified by its text."""
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    low = text.lower()
    if any(k in low for k in ("rate limit", "rate_limit", "too many", "concurren", "capacity",
                              "overloaded", "resource_exhausted", "queue")):
        return ApiError(429, text)
    if any(k in low for k in ("unauthor", "api key", "api_key", "auth_error", "billing", "payment",
                              "insufficient", "credit")):
        return ApiError(401, text)
    return ApiError(default, text)


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


async def send_paced(send, pcm, chunk_ms=100, speed=1.0):
    """Send PCM16 in chunk_ms pieces on an absolute real-time schedule (speed 0 = no pacing)."""
    step = int(SR * 2 * chunk_ms / 1000)
    t0 = time.perf_counter()
    for i, off in enumerate(range(0, len(pcm), step)):
        await send(pcm[off:off + step])
        if speed > 0:
            delay = t0 + (i + 1) * chunk_ms / 1000 / speed - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)


def stream_timeout(clip, speed, extra=60.0):
    return (clip.duration / speed if speed > 0 else clip.duration) + extra


async def run_session(sender, receiver):
    """Run a streaming session's sender and receiver concurrently; return the receiver's result.

    The receiver returns once the server has delivered the final transcript. An exception in
    either one cancels the other and is raised.
    """
    tasks = [asyncio.ensure_future(sender), asyncio.ensure_future(receiver)]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in tasks:
            if t.done() and t.exception():
                raise t.exception()
        return tasks[1].result()
    finally:
        for t in tasks:
            t.cancel()


async def streaming(session, clip, speed, extra=60.0, close_codes=None):
    """Run one streaming session with a time budget; map websocket failures to ApiError.

    close_codes maps a provider's close codes to a status (e.g. {4001: 401} aborts the run);
    any other abnormal close is retried as 503.
    """
    try:
        return await asyncio.wait_for(session, stream_timeout(clip, speed, extra))
    except websockets.InvalidStatus as e:  # handshake rejected (auth, rate limit, bad params)
        r = e.response
        raise ApiError(r.status_code, (r.body or b"")[:300].decode("utf-8", "replace"),
                       retry_after(r.headers)) from e
    except websockets.ConnectionClosedError as e:  # closed mid-session
        code, reason = (e.rcvd.code, e.rcvd.reason) if e.rcvd else (None, str(e))
        raise ApiError((close_codes or {}).get(code, 503), f"socket closed {code}: {reason}") from e


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
    main(lambda args: Model(args.speed), "xai_grok-voice-transcribe-2.0-streaming", concurrency=20)
