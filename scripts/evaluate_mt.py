import os
import json
import argparse
import sacrebleu
from transformers import MarianMTModel, MarianTokenizer
from bert_score import score

def compute_bleu(dataset, metadata):
    base = f"data/{dataset}"
    translation_dir = f"{base}/translations"
    reference_dir = f"{base}/reference_translations"
    results_path = f"{base}/mt_evaluation.json"

    results = []
    total = 0
    evaluated = 0

    for item in metadata:
        video_id = item["id"]
        translation_path = os.path.join(translation_dir, f"{video_id}_es.txt")
        reference_path = os.path.join(reference_dir, f"{video_id}_es.txt")
        total += 1
        if not os.path.exists(translation_path) or not os.path.exists(reference_path):
            print(f"Skipping {video_id} (missing translation or reference)")
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
    print(f"\nEvaluated {evaluated}/{total} with BLEU. Results saved to {results_path}")

def compute_bertscore_backtranslation(dataset, metadata):
    base = f"data/{dataset}"
    asr_dir = os.path.join(base, "original_transcripts")
    mt_dir = os.path.join(base, "translations")
    bt_dir = os.path.join(base, "back_translations")
    output_file = os.path.join(base, "bertscore.json")

    os.makedirs(bt_dir, exist_ok=True)

    print("Loading back-translation model")
    model_name = "Helsinki-NLP/opus-mt-es-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    results = []
    total = 0
    evaluated = 0

    for item in metadata:
        video_id = item["id"]
        translation_path = os.path.join(mt_dir, f"{video_id}_es.txt")
        asr_path = os.path.join(asr_dir, f"{video_id}.txt")
        bt_path = os.path.join(bt_dir, f"{video_id}_bt.txt")

        total += 1
        if not os.path.exists(translation_path) or not os.path.exists(asr_path):
            print(f"Skipping {video_id} (missing ASR or translation)")
            continue

        with open(translation_path, "r", encoding="utf-8") as f:
            spanish = f.read().strip()
        with open(asr_path, "r", encoding="utf-8") as f:
            original_en = f.read().strip()

        sentences = spanish.split(". ")
        back_translated = []
        for sent in sentences:
            if sent.strip() == "":
                continue
            inputs = tokenizer.prepare_seq2seq_batch([sent], return_tensors="pt")
            output_tokens = model.generate(**inputs)
            en_sent = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            back_translated.append(en_sent)

        bt_text = ". ".join(back_translated)
        with open(bt_path, "w", encoding="utf-8") as f:
            f.write(bt_text)

        P, R, F1 = score([bt_text], [original_en], lang="en", model_type="bert-base-uncased")
        score_f1 = round(F1.mean().item(), 4)

        results.append({
            "id": video_id,
            "bertscore_f1": score_f1
        })
        evaluated += 1
        print(f"{video_id} → BERTScore = {score_f1}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluated {evaluated}/{total} with BERTScore. Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ted_talks')
    args = parser.parse_args()

    base = f"data/{args.dataset}"
    metadata_path = os.path.join(base, "metadata.json")
    reference_dir = os.path.join(base, "reference_translations")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if os.path.exists(reference_dir) and len(os.listdir(reference_dir)) > 0:
        print("Reference translations found so BLEU will be used for evaluation.")
        compute_bleu(args.dataset, metadata)
    else:
        print("No reference translations so back-translation + BERTScore will be used for evaluation.")
        compute_bertscore_backtranslation(args.dataset, metadata)

if __name__ == '__main__':
    main()
