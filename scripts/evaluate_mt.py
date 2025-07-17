import os
import sacrebleu
import json

translation_dir = "data/translations"
reference_dir = "data/reference_translations"
results_path = "data/mt_evaluation.json"

results = []

for fname in os.listdir(translation_dir):
    if not fname.endswith("_es.txt"):
        continue

    video_id = fname.replace("_es.txt", "")
    translation_path = os.path.join(translation_dir, fname)
    reference_path = os.path.join(reference_dir, f"{video_id}_es.txt")

    if not os.path.exists(reference_path):
        print(f"No reference translation for {video_id}, this video will be skipped")
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
    print(f"BLEU score for {video_id}: {bleu.score:.2f}")

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n Evaluation complete. Results saved to:", results_path)
