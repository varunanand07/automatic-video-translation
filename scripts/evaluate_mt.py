import os
import json
import argparse
import sacrebleu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ted_talks')
    parser.add_argument('--mt-model', type=str)
    args = parser.parse_args()

    base = f"data/{args.dataset}"
    metadata_path = os.path.join(base, "metadata.json")
    reference_dir = os.path.join(base, "reference_translations")
    translations_dir = os.path.join(base, "translations")
    results_path = os.path.join(base, "mt_evaluation.json")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if args.mt_model:
        models = [args.mt_model]
    else:
        models = [d for d in os.listdir(translations_dir) if os.path.isdir(os.path.join(translations_dir, d))]
    
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            if isinstance(existing_data, list):
                all_results = {}
            else:
                all_results = existing_data
    else:
        all_results = {}

    for model in models:
        translation_dir = os.path.join(translations_dir, model)
        
        if not os.path.exists(translation_dir):
            print(f"Translation directory not found {translation_dir}")
            continue

        model_results = []
        total = 0
        evaluated = 0

        for item in metadata:
            video_id = item["id"]
            translation_path = os.path.join(translation_dir, f"{video_id}_es.txt")
            reference_path = os.path.join(reference_dir, f"{video_id}_es.txt")
            total += 1
            
            if not os.path.exists(translation_path):
                print(f"Skipping {video_id} because translation is missing")
                continue
            if not os.path.exists(reference_path):
                print(f"Skipping {video_id} because reference translation is missing")
                continue

            with open(translation_path, "r", encoding="utf-8") as f:
                system_translation = f.read().strip()
            with open(reference_path, "r", encoding="utf-8") as f:
                reference_translation = f.read().strip()

            bleu = sacrebleu.corpus_bleu([system_translation], [[reference_translation]])
            model_results.append({
                "id": video_id,
                "title": item.get("title", ""),
                "bleu_score": round(bleu.score, 2)
            })
            evaluated += 1

        all_results[model] = model_results

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_path}")

if __name__ == '__main__':
    main()
