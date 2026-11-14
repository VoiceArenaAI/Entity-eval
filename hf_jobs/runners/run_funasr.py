"""Open ASR Leaderboard-style eval runner for FunASR models.

Supported models and the settings used, taken from each model card's own
long-audio (VAD) recipe:

FunAudioLLM/Fun-ASR-MLT-Nano-2512 (https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512)
    AutoModel(model=..., hub="hf", vad_model="fsmn-vad",
              vad_kwargs={"max_single_segment_time": 30000})
    generate(cache={}, batch_size=1, language="英文", itn=True,
             llm_kwargs={"do_sample": False})
    The language hint is inserted verbatim into the model's Chinese prompt
    ("语音转写成{language}"), so English is "英文" (the card lists the languages by
    their Chinese names). The card's trust_remote_code/remote_code="./model.py"
    point at a file from the Fun-ASR GitHub repo; funasr>=1.4.1 ships the same
    FunASRNano class, so they are not needed.

FunAudioLLM/SenseVoiceSmall (https://huggingface.co/FunAudioLLM/SenseVoiceSmall)
    AutoModel(model=..., hub="hf", vad_model="fsmn-vad",
              vad_kwargs={"max_single_segment_time": 30000})
    generate(cache={}, language="en", use_itn=True, batch_size_s=60,
             merge_vad=True, merge_length_s=15)
    The output keeps SenseVoice's rich-transcription tags (language, emotion,
    audio event); they are converted with the card's own
    funasr.utils.postprocess_utils.rich_transcription_postprocess.

With VAD, funasr decodes the clips of one generate() call one after another
and batches the VAD segments within each clip, so --batch_size only sets how
many clips share one timed call. Per-sample time is the wall time of that call
divided by the number of clips, as in transformers/run_eval.py.

Install (image hf.co/spaces/hf-audio/open-asr-leaderboard-transformers, which
already has python 3.12, torch/torchaudio 2.8.0 and datasets 3.6.0; run with
PYTHONPATH=/app for normalizer/):
    pip install funasr==1.4.26
"""

import argparse
import os
import time

import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from normalizer import data_utils, eval_utils
from tqdm import tqdm

VAD = {"vad_model": "fsmn-vad", "vad_kwargs": {"max_single_segment_time": 30000}}

MODELS = {
    "FunAudioLLM/Fun-ASR-MLT-Nano-2512": {
        "generate": {"batch_size": 1, "language": "英文", "itn": True, "llm_kwargs": {"do_sample": False}},
        "postprocess": str.strip,
    },
    "FunAudioLLM/SenseVoiceSmall": {
        "generate": {"language": "en", "use_itn": True, "batch_size_s": 60, "merge_vad": True, "merge_length_s": 15},
        "postprocess": lambda text: rich_transcription_postprocess(text).strip(),
    },
}


def main(args):
    config = MODELS[args.model_id]
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    model = AutoModel(
        model=args.model_id, hub="hf", device=device, disable_update=True, disable_pbar=True, log_level="WARNING", **VAD
    )

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)

    def transcribe(audios):
        return [config["postprocess"](r["text"]) for r in model.generate(input=audios, cache={}, **config["generate"])]

    def benchmark(batch):
        audios = [a["array"].astype("float32") for a in batch["audio"]]
        sync()
        start = time.time()
        predictions = transcribe(audios)
        sync()
        runtime = time.time() - start

        batch["audio_length_s"] = [len(a) / 16_000 for a in audios]
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(batch, len(audios))
        batch["transcription_time_s"] = [runtime / len(audios)] * len(audios)
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
    dataset = data_utils.prepare_data(dataset, sampling_rate=16_000)

    if args.warmup_steps > 0:
        for sample in dataset.take(args.warmup_steps):
            transcribe([sample["audio"]["array"].astype("float32")])

    dataset = dataset.map(benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"])

    results = {k: [] for k in ("audio_length_s", "transcription_time_s", "predictions", "references", "audio_filepath")}
    for result in tqdm(iter(dataset), desc="Samples..."):
        for key in results:
            results[key].append(result[key])

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
    parser.add_argument("--model_id", type=str, required=True, choices=sorted(MODELS))
    parser.add_argument("--dataset_path", type=str, default="hf-audio/open-asr-leaderboard")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset config name.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0 for the first GPU and so on.")
    parser.add_argument("--batch_size", type=int, default=16, help="Clips per timed generate() call.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="-1 or unset for all samples.")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=2, help="Untimed clips decoded before the timed run.")
    args = parser.parse_args()

    print("*" * 100)
    print(f"Evaluating {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)
    main(args)
