import os
import json
import argparse
import random
import math
from statistics import mean, pstdev
import sacrebleu

def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=42):
    if not values:
        return (math.nan, math.nan)
    rnd = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(n_boot):
        sample = [values[rnd.randrange(n)] for _ in range(n)]
        boots.append(mean(sample))
    lo = sorted(boots)[int((alpha/2)*n_boot)]
    hi = sorted(boots)[int((1-alpha/2)*n_boot)]
    return lo, hi

def detect_ref_type(dataset_name):
    return "human" if dataset_name == "ted_talks" else "synthetic"

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
    summary_path = os.path.join(base, "mt_evaluation_summary.json")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if args.mt_model:
        models = [args.mt_model]
    else:
        models = [d for d in os.listdir(translations_dir) if os.path.isdir(os.path.join(translations_dir, d))]

    all_results = {}
    ref_type = detect_ref_type(os.path.basename(base))

    for model in models:
        print(f"\nEvaluating model: {model}")
        translation_dir = os.path.join(translations_dir, model)
        if not os.path.exists(translation_dir):
            print(f"Translation folder not found: {translation_dir}")
            continue

        model_results = []
        total = 0
        evaluated = 0
        bleu_scores, chrf_scores, ter_scores = [], [], []

        for item in metadata:
            vid = item["id"]
            total += 1
            sys_path = os.path.join(translation_dir, f"{vid}_es.txt")
            ref_path = os.path.join(reference_dir, f"{vid}_es.txt")

            if not os.path.exists(sys_path) or not os.path.exists(ref_path):
                if not os.path.exists(sys_path):
                    print(f"System translation not found for {vid}")
                if not os.path.exists(ref_path):
                    print(f"Reference translation not found for {vid}")
                continue

            with open(sys_path, "r", encoding="utf-8") as f:
                sys_text = f.read().strip()
            with open(ref_path, "r", encoding="utf-8") as f:
                ref_text = f.read().strip()

            bleu = sacrebleu.corpus_bleu([sys_text], [[ref_text]])
            chrf = sacrebleu.corpus_chrf([sys_text], [[ref_text]], word_order=2)
            ter = sacrebleu.corpus_ter([sys_text], [[ref_text]])

            entry = {
                "id": vid,
                "title": item.get("title", ""),
                "bleu_score": round(bleu.score, 2),
                "chrf_score": round(chrf.score, 2),
                "ter_score": round(ter.score, 2),
            }

            bleu_scores.append(bleu.score)
            chrf_scores.append(chrf.score)
            ter_scores.append(ter.score)

            model_results.append(entry)
            evaluated += 1

        all_results[model] = model_results
        print(f"{model}: evaluated {evaluated}/{total} videos")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    summary = {"dataset": os.path.basename(base), "reference_type": ref_type, "models": {}}
    for model, rows in all_results.items():
        bs = [r["bleu_score"] for r in rows]
        cs = [r["chrf_score"] for r in rows]
        ts = [r["ter_score"]  for r in rows]

        def stats(arr):
            if not arr:
                return {"mean": None, "ci95": [None, None], "n": 0, "std": None}
            lo, hi = bootstrap_ci(arr)
            return {
                "mean": round(mean(arr), 2),
                "ci95": [round(lo, 2), round(hi, 2)],
                "n": len(arr),
                "std": round(pstdev(arr), 2)
            }

        summary["models"][model] = {
            "BLEU": stats(bs),
            "chrF++": stats(cs),
            "TER": stats(ts)
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results for each video to {results_path}")
    print(f"Saved summary with CIs to   {summary_path}")

if __name__ == '__main__':
    main()
