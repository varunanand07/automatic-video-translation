import os
import json
import argparse
from jiwer import wer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ted_talks')
    parser.add_argument('--model', type=str, default='whisper')
    args = parser.parse_args()

    dataset = args.dataset
    model = args.model
    base = f"data/{dataset}"
    original_dir = os.path.join(base, "original_transcripts")
    asr_dir = os.path.join(base, f"asr_transcripts/{model}")
    metadata_path = os.path.join(base, "metadata.json")
    output_path = os.path.join(base, "asr_evaluation.json")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            previous_data = json.load(f)
        existing_results = {}
        for entry in previous_data:
            wer_value = entry.get("wer")
            if isinstance(wer_value, float):
                entry["wer"] = {"whisper": wer_value}
            existing_results[entry["id"]] = entry
    else:
        existing_results = {}

    updated_count = 0
    skipped = 0

    for item in metadata:
        video_id = item["id"]
        title = item.get("title", "")
        original_file = os.path.join(original_dir, f"{video_id}.txt")
        asr_file = os.path.join(asr_dir, f"{video_id}.txt")

        if not os.path.exists(original_file) or not os.path.exists(asr_file):
            print(f"Skipping {video_id} due to missing transcript")
            skipped += 1
            continue

        with open(original_file, "r", encoding="utf-8") as f:
            reference = f.read()
        with open(asr_file, "r", encoding="utf-8") as f:
            hypothesis = f.read()

        error = round(wer(reference, hypothesis), 3)

        entry = existing_results.get(video_id, {"id": video_id, "title": title, "wer": {}})
        if isinstance(entry["wer"], float):
            entry["wer"] = {"whisper": entry["wer"]}
        entry["wer"][model] = error
        existing_results[video_id] = entry

        updated_count += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(list(existing_results.values()), f, indent=2)

    print(f"Results saved to: {output_path}")

if __name__ == '__main__':
    main()
