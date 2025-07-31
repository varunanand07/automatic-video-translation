import os
import argparse
from transformers import (
    MarianMTModel, MarianTokenizer,
    MBartForConditionalGeneration, MBart50TokenizerFast,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
import torch

def load_model_and_tokenizer(mt_model, device="cpu"):
    if mt_model == "marian":
        model_name = "Helsinki-NLP/opus-mt-en-es"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)
        if device == "cuda":
            model = model.half()
        def prepare_batch(sents):
            return tokenizer.prepare_seq2seq_batch(sents, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        return tokenizer, model, prepare_batch

    elif mt_model == "mbart":
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
        if device == "cuda":
            model = model.half()
        tokenizer.src_lang = "en_XX"
        def prepare_batch(sents):
            inputs = tokenizer(sents, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["forced_bos_token_id"] = tokenizer.lang_code_to_id["es_XX"]
            return inputs
        return tokenizer, model, prepare_batch

    elif mt_model == "nllb":
        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        if device == "cuda":
            model = model.half()
        tokenizer.src_lang = "eng_Latn"
        def prepare_batch(sents):
            inputs = tokenizer(sents, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids("spa_Latn")
            return inputs
        return tokenizer, model, prepare_batch

    else:
        raise ValueError(f"Unknown MT model: {mt_model}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ted_talks')
    parser.add_argument('--mt-model', type=str, default='marian')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    asr_model = "whisper" 
    base = f"data/{args.dataset}"
    asr_dir = os.path.join(base, f"asr_transcripts/{asr_model}")
    translation_dir = os.path.join(base, f"translations/{args.mt_model}")
    os.makedirs(translation_dir, exist_ok=True)

    tokenizer, model, prepare_batch = load_model_and_tokenizer(args.mt_model, device)
    model.eval()

    processed = 0
    for fname in os.listdir(asr_dir):
        if not fname.endswith(".txt"):
            continue
        video_id = fname.replace(".txt", "")
        input_path = os.path.join(asr_dir, fname)
        output_path = os.path.join(translation_dir, f"{video_id}_es.txt")
        if os.path.exists(output_path):
            print(f"Already translated: {fname}")
            continue
        
        with open(input_path, "r", encoding="utf-8") as f:
            english_text = f.read().strip()
        sentences = [s.strip() for s in english_text.split(". ") if s.strip()]
        translated = []
        total_sentences = len(sentences)
        
        for i in range(0, len(sentences), args.batch_size):
            batch = sentences[i:i+args.batch_size]
            batch_num = i // args.batch_size + 1
            total_batches = (len(sentences) + args.batch_size - 1) // args.batch_size
            
            try:
                inputs = prepare_batch(batch)
                with torch.no_grad():
                    translated_tokens = model.generate(
                        **inputs, 
                        max_length=256,      
                        num_beams=1,         
                        do_sample=False,     
                        early_stopping=True
                    )
                batch_translations = []
                for j, tokens in enumerate(translated_tokens):
                    translation = tokenizer.decode(tokens, skip_special_tokens=True)
                    batch_translations.append(translation)
                
                translated.extend(batch_translations)
                print(f"Completed batch {batch_num}/{total_batches}")
                
            except Exception as e:
                print(f"Failed to translate batch {batch_num}: {e}")
                translated.extend([""] * len(batch))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(". ".join(translated))
        print(f"Translated {fname}: {output_path}")
        processed += 1

    print(f"\nTranslated {processed} files using {args.mt_model} model (ASR: whisper)")

if __name__ == '__main__':
    main()
