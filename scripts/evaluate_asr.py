import os
from jiwer import wer
import json

original_dir = "data/original_transcripts"
asr_dir = "data/asr_transcripts"
metadata_path = "data/metadata.json"

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

results = []

for item in metadata:
    video_id = item["id"]
    original_file = f"{original_dir}/{video_id}.txt"
    asr_file = f"{asr_dir}/{video_id}.txt"

    if not os.path.exists(original_file):
        print(f"Original transcript not found for {video_id}")
        continue
    if not os.path.exists(asr_file):
        print(f"ASR transcript not found for {video_id}")
        continue

    with open(original_file, "r", encoding="utf-8") as f:
        reference = f.read()
    with open(asr_file, "r", encoding="utf-8") as f:
        hypothesis = f.read()

    error = wer(reference, hypothesis)

    results.append({
        "id": video_id,
        "title": item.get("title", ""),
        "wer": round(error, 3)
    })

print("\nWord Error Rates:")
for r in results:
    print(f"{r['title']} ({r['id']}): WER = {r['wer']}")

with open("data/asr_evaluation.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n Results saved to data/asr_evaluation.json")
