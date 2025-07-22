import os
import re

DATASETS = ["ted_talks", "podcasts", "lectures", "youtube_shorts"]
BASE_DIR = "data"
TARGET_FOLDER = "original_transcripts"

TO_LOWER = True
REMOVE_NON_SPEECH = True

NON_SPEECH_RE = re.compile(r'\[.*?\]|\(.*?\)', re.IGNORECASE)
MULTI_SPACES = re.compile(r'\s+')

def clean_text(text):
    if REMOVE_NON_SPEECH:
        text = NON_SPEECH_RE.sub('', text)

    text = MULTI_SPACES.sub(' ', text)

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if TO_LOWER:
        lines = [line.lower() for line in lines]

    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            unique_lines.append(line)
            seen.add(line)

    return "\n".join(unique_lines)

for dataset in DATASETS:
    dir_path = os.path.join(BASE_DIR, dataset, TARGET_FOLDER)
    print(f"\nCleaning transcripts in dataset: {dataset}")
    for fname in os.listdir(dir_path):
        if not fname.endswith(".txt"):
            continue

        file_path = os.path.join(dir_path, fname)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            cleaned_text = clean_text(raw_text)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            print(f"Cleaned: {fname}")

        except Exception as e:
            print(f"Failed to clean {fname}: {e}")
