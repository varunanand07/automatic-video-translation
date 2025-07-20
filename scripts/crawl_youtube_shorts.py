import yt_dlp
import os
import json
import webvtt
import time
import random

dataset_name = "youtube_shorts"
base_dir = f"data/{dataset_name}"
transcript_dir = f"{base_dir}/original_transcripts"
reference_dir = f"{base_dir}/reference_translations"
metadata_file = f"{base_dir}/metadata.json"
os.makedirs(transcript_dir, exist_ok=True)
os.makedirs(reference_dir, exist_ok=True)

video_urls = [
    "https://www.youtube.com/shorts/ErpdSOfPa-Q"
]

metadata = []

ydl_opts = {
    'skip_download': True,
    'writesubtitles': True,
    'writeautomaticsub': True,
    'subtitleslangs': ['en', 'es'],
    'subtitlesformat': 'vtt',
    'outtmpl': f'{transcript_dir}/%(id)s.%(ext)s',
    'quiet': True
}

def vtt_to_txt(vtt_path, txt_path):
    try:
        lines = [caption.text.strip() for caption in webvtt.read(vtt_path)]
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    except Exception as e:
        print(f"Error processing {vtt_path}: {e}")

def download_with_retry(ydl, url, max_retries=3):
    for attempt in range(max_retries):
        try:
            info = ydl.extract_info(url, download=True)
            return info
        except Exception as e:
            if "429" in str(e):
                wait = (2 ** attempt) + random.uniform(1, 5)
                print(f"Rate limited. Retrying in {wait:.1f}s")
                time.sleep(wait)
            else:
                raise e
    return None

for i, url in enumerate(video_urls):
    try:
        print(f"\nProcessing youtube short {i+1}/{len(video_urls)}: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = download_with_retry(ydl, url)
            if not info: continue

            video_id = info['id']

            vtt_en = f"{transcript_dir}/{video_id}.en.vtt"
            txt_en = f"{transcript_dir}/{video_id}.txt"
            if os.path.exists(vtt_en):
                vtt_to_txt(vtt_en, txt_en)
                os.remove(vtt_en)

            vtt_es = f"{transcript_dir}/{video_id}.es.vtt"
            txt_es = f"{reference_dir}/{video_id}_es.txt"
            if os.path.exists(vtt_es):
                vtt_to_txt(vtt_es, txt_es)
                os.remove(vtt_es)

            metadata.append({
                'id': video_id,
                'title': info.get('title'),
                'url': url,
                'duration': info.get('duration'),
                'subtitles_available': os.path.exists(txt_en)
            })

    except Exception as e:
        print(f"Error on {url}: {e}")
        continue

with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print("Youtube shorts crawling complete.")