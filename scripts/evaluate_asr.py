import os
import json
import argparse
from jiwer import wer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ted_talks')
    args = parser.parse_args()

    base = f"data/{args.dataset}"
    original_dir = f"{base}/original_transcripts"
    asr_dir = f"{base}/asr_transcripts"
    metadata_path = f"{base}/metadata.json"
    output_path = f"{base}/asr_evaluation.json"

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    results = []
    total = 0
    evaluated = 0
    for item in metadata:
        video_id = item["id"]
        original_file = f"{original_dir}/{video_id}.txt"
        asr_file = f"{asr_dir}/{video_id}.txt"
        total += 1
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
        evaluated += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluated {evaluated}/{total} items. Results saved to {output_path}")

if __name__ == '__main__':
    main()
