import os
from transformers import MarianMTModel, MarianTokenizer

asr_dir = "data/asr_transcripts"
translation_dir = "data/translations"
os.makedirs(translation_dir, exist_ok=True)

model_name = "Helsinki-NLP/opus-mt-en-es"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

for fname in os.listdir(asr_dir):
    if not fname.endswith(".txt"):
        continue

    video_id = fname.replace(".txt", "")
    input_path = os.path.join(asr_dir, fname)
    output_path = os.path.join(translation_dir, f"{video_id}_es.txt")

    if os.path.exists(output_path):
        print(f"This has already been translated: {fname}")
        continue

    with open(input_path, "r", encoding="utf-8") as f:
        english_text = f.read().strip()

    sentences = english_text.split(". ")
    translated = []

    for sent in sentences:
        if sent.strip() == "":
            continue
        inputs = tokenizer.prepare_seq2seq_batch([sent], return_tensors="pt", padding=True)
        translated_tokens = model.generate(**inputs)
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        translated.append(translated_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(". ".join(translated))
    
    print(f"Translated {fname}. Results saved to {output_path}")
