import os
import sacrebleu
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ted_talks')
    args = parser.parse_args()

    base = f"data/{args.dataset}"
    translation_dir = f"{base}/translations"
    reference_dir = f"{base}/reference_translations"
    results_path = f"{base}/mt_evaluation.json"
    metadata_path = f"{base}/metadata.json"

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    results = []
    total = 0
    evaluated = 0
    for item in metadata:
        video_id = item["id"]
        translation_path = os.path.join(translation_dir, f"{video_id}_es.txt")
        reference_path = os.path.join(reference_dir, f"{video_id}_es.txt")
        total += 1
        if not os.path.exists(translation_path):
            print(f"No system translation for {video_id}, skipping")
            continue
        if not os.path.exists(reference_path):
            print(f"No reference translation for {video_id}, skipping")
            continue
        with open(translation_path, "r", encoding="utf-8") as f:
            system_translation = f.read().strip()
        with open(reference_path, "r", encoding="utf-8") as f:
            reference_translation = f.read().strip()
        bleu = sacrebleu.corpus_bleu([system_translation], [[reference_translation]])
        results.append({
            "id": video_id,
            "bleu_score": round(bleu.score, 2)
        })
        evaluated += 1
        print(f"BLEU score for {video_id}: {bleu.score:.2f}")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluated {evaluated}/{total} items. Results saved to {results_path}")

if __name__ == '__main__':
    main()