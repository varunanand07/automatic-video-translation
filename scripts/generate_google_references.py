import os
import json
import argparse
import time
import random
from googletrans import Translator

def translate_with_retry(translator, text, retries=3):
    for i in range(retries):
        try:
            time.sleep(random.uniform(1, 2))
            result = translator.translate(text, src='en', dest='es')
            if result and result.text:
                return result.text
        except Exception as e:
            if i < retries - 1:
                time.sleep((2 ** i) + random.uniform(1, 3))
            else:
                raise e
    return None

def chunk_text(text, max_len=4000):
    sentences = text.split('. ')
    chunks, current = [], ""
    for sentence in sentences:
        if len(current + sentence) + 2 <= max_len:
            current += sentence + ". "
        else:
            if current:
                chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--asr-model', default='whisper')
    args = parser.parse_args()

    base = f"data/{args.dataset}"
    asr_dir = f"{base}/asr_transcripts/{args.asr_model}"
    ref_dir = f"{base}/reference_translations"
    
    os.makedirs(ref_dir, exist_ok=True)
    translator = Translator()
    
    with open(f"{base}/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    translated = 0
    for item in metadata:
        video_id = item["id"]
        input_path = f"{asr_dir}/{video_id}.txt"
        output_path = f"{ref_dir}/{video_id}_es.txt"

        if not os.path.exists(input_path):
            continue
        if os.path.exists(output_path):
            translated += 1
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            english = f.read().strip()
        
        if not english:
            continue
            
        try:
            if len(english) > 4000:
                chunks = chunk_text(english)
                translations = [translate_with_retry(translator, chunk) for chunk in chunks]
                if all(translations):
                    final_translation = " ".join(translations)
                else:
                    continue
            else:
                final_translation = translate_with_retry(translator, english)
            
            if final_translation:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(final_translation)
                translated += 1
                
        except Exception:
            continue
        
        time.sleep(random.uniform(2, 4))

    print(f"{translated} reference translation files saved to {ref_dir}")

if __name__ == '__main__':
    main()