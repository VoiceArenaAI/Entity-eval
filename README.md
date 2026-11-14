# Entity benchmark harness

Transcribes a set of accented-English clips with 56 speech-to-text APIs and 33 open-source models under identical
conditions, and saves one raw transcript per model and clip. Scoring is done separately.

## How it works

1. **Audio.** Each clip is downloaded once, converted to 16 kHz mono 16-bit PCM with soxr and cached in
   `audio_cache/`. Every model receives these exact samples.
2. **API models** (`api/`). One standalone script per model. Only the model, English and the audio format are set;
   everything else is the provider's default. Batch models get the WAV in one request. Streaming models get raw PCM16
   over a WebSocket at real time, and the transcript is the provider's final segments. The docstring at the top of
   each script gives the endpoint and the exact request.
3. **Open-source models** (`hf_jobs/`). Run on Hugging Face Jobs with the Open ASR Leaderboard's own Space images,
   eval scripts and normalizer (commit `859159c`), using the arguments from its submit scripts
   (`hf_jobs/models.json`). The audio goes in through a private Hub dataset; the scripts refuse to run if the
   dataset or the results bucket is public.

Every script resumes where it stopped, retries transient errors with back-off, stops on authentication or billing
errors, and writes each clip to disk as soon as it finishes.

## Requirements

- [uv](https://docs.astral.sh/uv/). Each script lists its dependencies at the top; `uv run` installs them.
- API keys in `.env` (copy `.env.example`). Only the keys for the models you run are needed.
- For `hf_jobs/`: a Hugging Face token with Jobs access (`HF_TOKEN` in `.env`) and a clone of
  [open_asr_leaderboard](https://github.com/huggingface/open_asr_leaderboard).

## Data

`data/prompts.csv` (not included), one row per sentence: `sentence_id`, `domain`, `text` (the reference), and one
recording URL per accent: `nigerian_audio_url`, `filipino_audio_url`, `indian_audio_url`, `east_african_audio_url`,
`mexican_audio_url`, `portuguese_audio_url`, `us_audio_url`. An empty URL leaves that clip out. See
[`prompts.example.csv`](prompts.example.csv) for the format. Use `--data` to point at another file.

## Run

```sh
cp .env.example .env                                   # add your keys

uv run api/deepgram_nova_3.py --limit 2                # smoke test: 2 sentences x 7 accents
./run_all.sh                                           # every API model, one process each (logs/)
uv run check_results.py                                # flag failed, empty or truncated clips

git clone https://github.com/huggingface/open_asr_leaderboard && git -C open_asr_leaderboard checkout 859159c
uv run hf_jobs/make_dataset.py --repo <ns>/<private-dataset>      # uses audio_cache/ from any API run
uv run hf_jobs/run_jobs.py launch --dataset <ns>/<private-dataset> --bucket <ns>/<private-bucket> \
    --leaderboard open_asr_leaderboard --limit 2 --configs us     # smoke test
uv run hf_jobs/run_jobs.py launch --dataset <ns>/<private-dataset> --bucket <ns>/<private-bucket> \
    --leaderboard open_asr_leaderboard
uv run hf_jobs/run_jobs.py status --bucket <ns>/<private-bucket>
uv run hf_jobs/collect.py --bucket <ns>/<private-bucket>         # download and verify

uv run combine.py --results results results_oss --out all_models.csv   # one column per model
```

API scripts also take `--accents`, `--concurrency` and `--retry-failed`. To resume, rerun the same command.

## Output

`results/<model>/<accent>.jsonl` and `.csv` for API models, `results_oss/<model>/<accent>.jsonl` and `.csv` for
open-source models, one record per clip: `sentence_id`, `domain`, `accent`, `audio_filepath`, `text` (reference),
`pred_text` (raw transcript), `duration`, `time`, `status` and `error`. API records also keep `attempts`,
`retry_errors` and the provider's `response` metadata.

## API models

| Provider | Batch | Streaming |
| --- | --- | --- |
| Amazon | `aws_transcribe`<br>`aws_transcribe_medical` | `aws_transcribe_medical_streaming`<br>`aws_transcribe_streaming` |
| AssemblyAI | `assemblyai_universal_3_6_pro` ¹ | `assemblyai_universal_3_6_pro_realtime` |
| Cartesia |  | `cartesia_ink_2` |
| Deepgram | `deepgram_nova_3` ¹ | `deepgram_flux_english`<br>`deepgram_flux_multilingual`<br>`deepgram_nova_3_streaming` |
| ElevenLabs | `elevenlabs_scribe_v2_medical` ¹<br>`elevenlabs_scribe_v3` | `elevenlabs_scribe_v3_live` |
| Fish Audio | `fish_transcribe_1_pro` ¹ | `fish_transcribe_stream_1` |
| Google | `google_chirp_3` ¹<br>`google_gemini_3_5_transcribe` ¹ | `google_gemini_3_5_transcribe_live` |
| Gradium |  | `gradium_stt` |
| Inworld |  | `inworld_stt_1` |
| Meta | `meta_muse_voice_transcribe_1` ¹ | `meta_muse_voice_transcribe_realtime` |
| Microsoft | `microsoft_mai_transcribe_2` ¹ |  |
| Mistral | `mistral_voxtral_mini_transcribe_3` ¹<br>`mistral_voxtral_small_transcribe_3` ¹ | `mistral_voxtral_mini_transcribe_realtime_3` |
| Modulate |  | `modulate_velma_2_stt_streaming_english_v2` |
| Murf |  | `murf_finch_1` |
| Omi Health | `omi_general_1`<br>`omi_medical_1`<br>`omi_medical_1_flash`<br>`omi_medical_edge_1` | `omi_general_1_live`<br>`omi_medical_1_live` |
| OpenAI | `openai_gpt_4o_mini_transcribe` ¹<br>`openai_gpt_transcribe` ¹ | `openai_gpt_live_transcribe`<br>`openai_gpt_realtime_whisper` |
| Reson8 | `reson8_resonant_1`<br>`reson8_rumble_2`<br>`reson8_rumble_2_flash` | `reson8_resonant_1_realtime`<br>`reson8_rumble_2_realtime` |
| Rime |  | `rime_stt_streaming` |
| Smallest.ai | `smallest_pulse_pro` |  |
| Soniox | `soniox_stt_async_v6` | `soniox_stt_rt_v6` |
| Speechmatics | `speechmatics_enhanced`<br>`speechmatics_melia_1`<br>`speechmatics_oak_1` | `speechmatics_enhanced_realtime`<br>`speechmatics_linden_1` |
| xAI | `xai_grok_voice_transcribe_2` | `xai_grok_voice_transcribe_2_streaming` |
| Zoom | `zoom_scribe_v2_1_pro` |  |

¹ called through OpenRouter's speech-to-text API.

## References

- Hugging Face, [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) and its evaluation
  code, [huggingface/open_asr_leaderboard](https://github.com/huggingface/open_asr_leaderboard) (commit `859159c`):
  the Space images, eval scripts, normalizer and model arguments used in `hf_jobs/`.
- Each provider's API documentation, linked in the docstring of every script in `api/`.

## License

MIT, see [LICENSE](LICENSE).
