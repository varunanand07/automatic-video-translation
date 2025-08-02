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
        print(f"\nEvaluating model: {model}")
        translation_dir = os.path.join(translations_dir, model)
        
        if not os.path.exists(translation_dir):
            print(f"Translation folder not found: {translation_dir}")
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
                continue
            if not os.path.exists(reference_path):
                print(f"Skipping {video_id} as reference translation is missing at {reference_path}")
                continue
            
            with open(translation_path, "r", encoding="utf-8") as f:
                system_translation = f.read().strip()
            with open(reference_path, "r", encoding="utf-8") as f:
                reference_translation = f.read().strip()

            bleu = sacrebleu.corpus_bleu([system_translation], [[reference_translation]])
            
            chrf = sacrebleu.corpus_chrf([system_translation], [[reference_translation]], word_order=2)
            
            ter = sacrebleu.corpus_ter([system_translation], [[reference_translation]])
            
            result = {
                "id": video_id,
                "title": item.get("title", ""),
                "bleu_score": round(bleu.score, 2),
                "chrf_score": round(chrf.score, 2),
                "ter_score": round(ter.score, 2)
            }
            
            model_results.append(result)
            evaluated += 1

        all_results[model] = model_results
        print(f"\n{model} evaluation complete. {evaluated}/{total} videos evaluated")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nEvaluation complete for {args.dataset}")

if __name__ == '__main__':
    main()
