"""Open ASR Leaderboard-style eval runner for Vosk (Kaldi) English on CPU.

Model: vosk-model-en-us-0.42-gigaspeech (https://alphacephei.com/vosk/models),
the newest and largest English model listed there. It is downloaded and
unzipped into --model_cache_dir on first use.

Each clip is converted to 16 kHz mono PCM16 and fed to a fresh KaldiRecognizer
in 4000-sample chunks. Vosk finalizes an utterance whenever it detects an
endpoint (AcceptWaveform returns True), so the transcript is the text of every
such Result() followed by FinalResult(), joined with spaces. Output is
lowercase words without punctuation.

The model is loaded once in the parent process and shared copy-on-write with a
fork-based ProcessPoolExecutor of --num_workers workers, so memory barely grows
with the worker count. Each batch of --batch_size clips is decoded in parallel, and
per-sample time is the batch wall time divided by the number of clips, as in
transformers/run_eval.py. RTFx is therefore the throughput of the whole pool.
The loaded model (graph plus rescoring LMs) peaks at about 10 GB of RAM.
On macOS, set VECLIB_MAXIMUM_THREADS=1: Accelerate's threaded BLAS crashes
in forked workers.

Install (python 3.12, Linux; run with PYTHONPATH pointing at the
open_asr_leaderboard checkout for normalizer/):
    pip install vosk==0.3.45 datasets==3.6.0 soundfile==0.14.0 librosa==1.0.1 \
        num2words==0.5.14 regex kaldialign==0.12.0
"""

import argparse
import http.client
import json
import multiprocessing
import os
import shutil
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from normalizer import data_utils, eval_utils
from tqdm import tqdm
from vosk import KaldiRecognizer, Model, SetLogLevel

SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 4000
MODEL_URL = "https://alphacephei.com/vosk/models/{}.zip"

_model = None  # set in the parent before the pool forks


def download(url, path, attempts=20):
    """Download with HTTP range resume; the model server often drops long transfers."""
    for _ in range(attempts):
        done = os.path.getsize(path) if os.path.exists(path) else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"})
        try:
            with urllib.request.urlopen(request) as response:
                if response.status != 206:  # range ignored: start over
                    done = 0
                total = done + int(response.headers["Content-Length"])
                with open(path, "ab" if done else "wb") as f:
                    shutil.copyfileobj(response, f)
        except (OSError, http.client.HTTPException) as e:
            print(f"Download interrupted ({e}), resuming")
            continue
        if os.path.getsize(path) == total:
            return
    raise RuntimeError(f"Could not download {url}")


def ensure_model(name, cache_dir):
    model_dir = os.path.join(cache_dir, name)
    if not os.path.isdir(model_dir):
        # Download and extract into a scratch dir first, so an interrupted run never leaves a partial model.
        tmp_dir = model_dir + ".partial"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir)
        zip_path = os.path.join(tmp_dir, name + ".zip")
        print(f"Downloading {MODEL_URL.format(name)} to {cache_dir}")
        download(MODEL_URL.format(name), zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_dir)
        os.rename(os.path.join(tmp_dir, name), model_dir)
        shutil.rmtree(tmp_dir)
    return model_dir


def transcribe(pcm16):
    recognizer = KaldiRecognizer(_model, SAMPLE_RATE)
    texts = []
    for start in range(0, len(pcm16), CHUNK_SAMPLES):
        if recognizer.AcceptWaveform(pcm16[start : start + CHUNK_SAMPLES].tobytes()):
            texts.append(json.loads(recognizer.Result())["text"])
    texts.append(json.loads(recognizer.FinalResult())["text"])
    return " ".join(t for t in texts if t)


def to_pcm16(array):
    return (np.clip(array, -1.0, 1.0) * 32767).astype(np.int16)


def main(args):
    global _model
    SetLogLevel(-1)
    _model = Model(ensure_model(args.model_id, args.model_cache_dir))
    pool = ProcessPoolExecutor(args.num_workers, mp_context=multiprocessing.get_context("fork"))

    def benchmark(batch):
        clips = [to_pcm16(a["array"]) for a in batch["audio"]]
        start = time.time()
        predictions = list(pool.map(transcribe, clips))
        runtime = time.time() - start

        batch["audio_length_s"] = [len(c) / SAMPLE_RATE for c in clips]
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(batch, len(clips))
        batch["transcription_time_s"] = [runtime / len(clips)] * len(clips)
        batch["predictions"] = predictions  # raw; normalization applied at scoring time
        batch["references"] = batch["original_text"]
        return batch

    dataset = data_utils.load_data(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dataset = data_utils.prepare_data(dataset, sampling_rate=SAMPLE_RATE)
    dataset = dataset.map(benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"])

    results = {k: [] for k in ("audio_length_s", "transcription_time_s", "predictions", "references", "audio_filepath")}
    for result in tqdm(iter(dataset), desc="Samples..."):
        for key in results:
            results[key].append(result[key])
    pool.shutdown()

    manifest_path = data_utils.write_manifest(
        results["references"],
        results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=results["audio_length_s"],
        transcription_time=results["transcription_time_s"],
        audio_filepaths=results["audio_filepath"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))
    score = eval_utils.score_manifest(eval_utils.read_manifest(manifest_path))
    print("WER:", score["wer"], "%", "RTFx:", score["rtfx"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="vosk-model-en-us-0.42-gigaspeech", help="Vosk model name.")
    parser.add_argument("--model_cache_dir", type=str, default=os.path.expanduser("~/.cache/vosk"))
    parser.add_argument("--dataset_path", type=str, default="hf-audio/open-asr-leaderboard")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset config name.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=-1, help="Unused; Vosk runs on CPU.")
    parser.add_argument("--batch_size", type=int, default=64, help="Clips decoded in parallel per timed batch.")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    parser.add_argument("--max_eval_samples", type=int, default=None, help="-1 or unset for all samples.")
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    print("*" * 100)
    print(f"Evaluating {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)
    main(args)
