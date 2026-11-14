"""Open ASR Leaderboard-style eval runner for moondream/parakeet-ultra.

Model card: https://huggingface.co/moondream/parakeet-ultra
(a post-trained nvidia/parakeet-tdt-0.6b-v3, same architecture and tokenizer).

The leaderboard's transformers/run_eval.py rejects this checkpoint
(ParakeetTDTConfig is not in the Transformers auto mappings), so this runner
uses the card's own inference path, Moondream's local Photon runtime:

    speech = md.photon("moondream/parakeet-ultra")
    speech.transcribe(audio=...)["text"]

Settings are the card's defaults, unchanged: no runtime options other than the
device, no decode settings (Parakeet TDT decodes greedily, temperature 0), and
the default timestamps="segment". Audio is passed as 16 kHz float PCM with
sample_rate=16000. Photon itself cuts long recordings at pauses found by the
model's voice-activity head into segments of at most 30 s and joins the text,
so full-length clips are transcribed without truncation.

The card documents no batch call. Photon batches concurrent requests into one
forward pass (up to 128 per pass on CUDA devices with >= 40 GiB, 64 on smaller
ones, 8 on CPU), so each --batch_size batch is submitted concurrently with
atranscribe. Per-sample time is the batch wall time divided by the batch size,
as in transformers/run_eval.py.

Install (python 3.12; keeps torch 2.8.0, since kestrel needs torch>=2.8; run
with PYTHONPATH pointing at the open_asr_leaderboard checkout for normalizer/):
    pip install moondream==2.6.1 kestrel==0.9.1
"""

import argparse
import asyncio
import os
import time

import moondream as md
from normalizer import data_utils, eval_utils
from tqdm import tqdm

SAMPLE_RATE = 16_000


async def transcribe_batch(speech, audios):
    results = await asyncio.gather(
        *(speech.atranscribe(audio=audio, sample_rate=SAMPLE_RATE) for audio in audios)
    )
    return [result["text"] for result in results]


def main(args):
    device = "cpu" if args.device < 0 else f"cuda:{args.device}"
    speech = md.photon(args.model_id, device=device)

    def benchmark(batch):
        audios = [audio["array"] for audio in batch["audio"]]
        start = time.perf_counter()
        predictions = asyncio.run(transcribe_batch(speech, audios))
        runtime = time.perf_counter() - start

        batch["audio_length_s"] = [len(audio) / SAMPLE_RATE for audio in audios]
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(batch, len(audios))
        batch["transcription_time_s"] = [runtime / len(audios)] * len(audios)
        batch["predictions"] = predictions  # raw; normalization applied at scoring time
        batch["references"] = batch["original_text"]
        return batch

    if args.warmup_steps > 0:
        warmup = data_utils.prepare_data(data_utils.load_data(args), sampling_rate=SAMPLE_RATE)
        num_warmup = args.warmup_steps * args.batch_size
        warmup = warmup.take(num_warmup) if args.streaming else warmup.select(range(min(num_warmup, len(warmup))))
        for _ in tqdm(warmup.map(benchmark, batch_size=args.batch_size, batched=True), desc="Warming up..."):
            continue

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
    speech.close()

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
    parser.add_argument("--model_id", type=str, default="moondream/parakeet-ultra")
    parser.add_argument("--dataset_path", type=str, default="hf-audio/open-asr-leaderboard")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset config name.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0 for the first GPU and so on.")
    parser.add_argument("--batch_size", type=int, default=64, help="Clips submitted concurrently per timed batch.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="-1 or unset for all samples.")
    parser.add_argument("--warmup_steps", type=int, default=2, help="Untimed batches run before evaluation.")
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    print("*" * 100)
    print(f"Evaluating {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)
    main(args)
