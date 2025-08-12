## Automatic Video Transcription

This repository contains a fully reproducible pipeline to:

- Collect or use provided English transcripts for YouTube videos
- Generate ASR transcripts (Whisper or AssemblyAI)
- Translate ASR transcripts to Spanish using three MT models (Marian, mBART, NLLB)
- Evaluate ASR (WER) and MT (BLEU, chrF, TER)
- Aggregate and visualize results

Datasets are already included under `data/` so the user can verify all results immediately without re-crawling or re-transcribing.

### Datasets included (size)
- `ted_talks` (10)
- `lectures` (4)
- `podcasts` (3)
- `youtube_shorts` (3)


## 1) Environment and prerequisites

### Requirements
- Python 3.10+ 
- ffmpeg (required by Whisper and yt-dlp)
- Internet (first run will download Hugging Face models)

### Install system deps
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`

### Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Python packages
```bash
pip install -r requirements.txt
```

Notes:
- Torch will use CPU by default, GPU is optional.
- First execution of translation scripts will download model weights (Marian, mBART, NLLB) to the local Hugging Face cache.

### Optional: AssemblyAI
To use AssemblyAI for ASR you will need an API key:
```bash
export ASSEMBLYAI_API_KEY=api_key
```


## 2) Repository structure

- `data/<dataset>/`
  - `original_transcripts/`: English text (from YouTube captions)
  - `audio/`: Downloaded audio (for Whisper usage)
  - `asr_transcripts/whisper/` and `asr_transcripts/assemblyai/`: ASR outputs
  - `reference_translations/`: Spanish references
  - `translations/{marian|mbart|nllb}/`: System translations
  - `metadata.json`: IDs, titles, URLs
  - `asr_evaluation.json`: WER per ASR system
  - `mt_evaluation.json`: MT metrics per MT system
  - `mt_evaluation_summary.json`: Dataset-level summary

- `scripts/`: All pipeline scripts (CLI usage below)
- `analysis/`: Aggregated CSVs (for visualization)
- `visualizations/`: Saved figures


## 3) One-command verification

Use the already-included data and metrics to regenerate aggregate CSVs and plots.

```bash
# 1) Aggregate all datasets → CSVs
python scripts/analyze_results.py

# 2) Create all figures
python scripts/visualize.py --asr-model whisper
```

You should now see PNGs in `visualizations/`:
- `bleu_by_model_dataset.png`
- `scatter_WER_BLEU_<dataset>.png` for all datasets
- `metric_correlations.png`

Compare these to the images in `visualizations/` to verify consistency.


## 4) Full pipeline verification

Below are commands to re-generate each component. You do not need to run them all to verify; use them selectively to validate any part.

### A) Crawl videos
Captions and metadata are already in `data/**/original_transcripts` and `metadata.json`. If you want to reproduce the crawl for a dataset:
```bash
python scripts/crawl_ted_talks.py
python scripts/crawl_lectures.py
python scripts/crawl_podcasts.py
python scripts/crawl_youtube_shorts.py
```
Notes:
- These use `yt_dlp` to fetch `.vtt` captions and convert to `.txt` using `webvtt`. YouTube may rate-limit, scripts retry with backoff.

### B) Transcribe audio with ASR (Whisper or AssemblyAI)
Audio and many transcripts are already present. To re-generate ASR transcripts:
```bash
# Whisper (default)
python scripts/transcribe_audio.py --dataset ted_talks --model whisper

# AssemblyAI (requires an AssemblyAI API key (Assembly_AI_API_KEY))
python scripts/transcribe_audio.py --dataset ted_talks --model assemblyai
```
Outputs go to `data/<dataset>/asr_transcripts/<model>/*.txt`.

### C) Translate ASR transcripts (Marian, mBART, NLLB)
```bash
python scripts/translate_asr.py --dataset ted_talks --mt-model marian
python scripts/translate_asr.py --dataset ted_talks --mt-model mbart
python scripts/translate_asr.py --dataset ted_talks --mt-model nllb
```
Outputs go to `data/<dataset>/translations/<mt-model>/*_es.txt`.

### D) Evaluate ASR (WER)
```bash
# Evaluate WER for Whisper outputs
python scripts/evaluate_asr.py --dataset ted_talks --model whisper

# Evaluate WER for AssemblyAI outputs (if present)
python scripts/evaluate_asr.py --dataset ted_talks --model assemblyai
```
Output: `data/<dataset>/asr_evaluation.json` (merged across ASR systems).

### E) Evaluate MT (BLEU, chrF, TER)
```bash
# Evaluate all MT systems found under translations/
python scripts/evaluate_mt.py --dataset ted_talks

# Or just one system
python scripts/evaluate_mt.py --dataset ted_talks --mt-model marian
```
Outputs:
- `data/<dataset>/mt_evaluation.json`
- `data/<dataset>/mt_evaluation_summary.json`

### F) Aggregate and visualize
```bash
# Aggregate all datasets into CSVs
python scripts/analyze_results.py

# Ensure analysis/ contains the latest CSVs for plotting
mkdir -p analysis
cp -f all_metrics_flat.csv analysis/
cp -f summary_by_ds_model.csv analysis/
cp -f correlations_WER_BLEU.csv analysis/
cp -f anova_model_x_dataset_bleu.csv analysis/

# Make figures
python scripts/visualize.py --asr-model whisper
```


## 5) Script reference

### scripts/transcribe_audio.py
- **args**: `--dataset {ted_talks|lectures|podcasts|youtube_shorts}`, `--model {whisper|assemblyai}`
- **reads**: `data/<dataset>/metadata.json` and downloads audio via `yt_dlp`
- **writes**: `data/<dataset>/audio/*.mp3`, `data/<dataset>/asr_transcripts/<model>/*.txt`

### scripts/translate_asr.py
- **args**: `--dataset <name>`, `--mt-model {marian|mbart|nllb}`, `--batch-size <int>`
- **models**:
  - marian: `Helsinki-NLP/opus-mt-en-es`
  - mbart: `facebook/mbart-large-50-many-to-many-mmt` (uses `en_XX` → `es_XX`)
  - nllb: `facebook/nllb-200-distilled-600M` (uses `eng_Latn` → `spa_Latn`)
- **writes**: `data/<dataset>/translations/<mt-model>/*_es.txt`

### scripts/evaluate_asr.py
- **args**: `--dataset <name>`, `--model {whisper|assemblyai}`
- **metric**: WER (via `jiwer`)
- **writes**: `data/<dataset>/asr_evaluation.json` (merges WER per ASR system)

### scripts/evaluate_mt.py
- **args**: `--dataset <name>`, `--mt-model <optional>`
- **metrics**: BLEU, chrF (word_order=2), TER (via `sacrebleu`)
- **writes**: `data/<dataset>/mt_evaluation.json`, `data/<dataset>/mt_evaluation_summary.json`

### scripts/analyze_results.py
- Aggregates per-dataset JSONs into CSVs, per model and dataset.
- **writes (to project root)**:
  - `all_metrics_flat.csv`
  - `summary_by_ds_model.csv`
  - `correlations_WER_BLEU.csv`
  - `anova_model_x_dataset_bleu.csv` (requires `statsmodels` for full ANOVA; otherwise a note is written)

### scripts/visualize.py
- **args**: `--asr-model whisper` (label for x-axis in WER plots)
- **reads**: `analysis/all_metrics_flat.csv`, `analysis/summary_by_ds_model.csv`
- **writes**: PNGs in `visualizations/`

### scripts/generate_google_references.py (optional)
- **args**: `--dataset <name>`, `--overwrite`
- Uses `deep-translator` (preferred) or `googletrans` to build synthetic Spanish references.

### scripts/clean_transcripts.py (optional)
- De-duplicates lines, strips bracketed non-speech, lowers text in `original_transcripts/` for all datasets.

## 6) Troubleshooting

- **ffmpeg not found**: Install ffmpeg (`brew install ffmpeg` or `apt-get install -y ffmpeg`).
- **Model download errors**: Ensure internet access; retry. Hugging Face caches in `~/.cache/huggingface`.
- **YouTube rate limits (HTTP 429)**: The crawl scripts back off automatically; consider rerunning later.
- **visualize.py cannot find CSVs**: The aggregator writes CSVs to the project root. Copy them to `analysis/` as shown above.
- **AssemblyAI 401**: Ensure `ASSEMBLYAI_API_KEY` is set.
- **Torch GPU**: If no GPU is available, scripts run on CPU by default.


## 7) License and citation

This repository is for research and evaluation. YouTube content belongs to the original creators. Models are from their respective providers (OpenAI Whisper, Helsinki-NLP, Meta, etc.) under their licenses.


