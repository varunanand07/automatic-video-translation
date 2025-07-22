import yt_dlp
import os
import json
import webvtt
import time
import random

dataset_name = "lectures"
base_dir = f"data/{dataset_name}"
transcript_dir = f"{base_dir}/original_transcripts"
metadata_file = f"{base_dir}/metadata.json"
os.makedirs(transcript_dir, exist_ok=True)

video_urls = [
    "https://www.youtube.com/watch?v=dxxEJbxuJOE", 
    "https://www.youtube.com/watch?v=Ps8jOj7diA0",
    "https://www.youtube.com/watch?v=1Pzk-UqilW4",
    "https://www.youtube.com/watch?v=oKtWYe9DxDo"
]

metadata = []

ydl_opts = {
    'skip_download': True,
    'writesubtitles': True,
    'writeautomaticsub': True,
    'subtitleslangs': ['en'],
    'subtitlesformat': 'vtt',
    'outtmpl': f'{transcript_dir}/%(id)s.%(ext)s',
    'quiet': True
}

def vtt_to_txt(vtt_path, txt_path):
    try:
        lines = []
        for caption in webvtt.read(vtt_path):
            for line in caption.text.split('\n'):
                line = line.strip()
                if line:
                    lines.append(line)
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
        print(f"\nProcessing lecture {i+1}/{len(video_urls)}: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = download_with_retry(ydl, url)
            if not info:
                continue
            video_id = info['id']
            vtt_en = f"{transcript_dir}/{video_id}.en.vtt"
            txt_en = f"{transcript_dir}/{video_id}.txt"
            transcript_found = False
            if os.path.exists(vtt_en):
                vtt_to_txt(vtt_en, txt_en)
                os.remove(vtt_en)
                transcript_found = True
                print(f"Transcript saved for {video_id}")
            else:
                print(f"No English transcript found for {video_id}")
            if transcript_found:
                metadata.append({
                    'id': video_id,
                    'title': info.get('title'),
                    'url': url,
                    'duration': info.get('duration'),
                    'transcript_available': True
                })
    except Exception as e:
        print(f"Error on {url}: {e}")
        continue

with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print("Lectures crawling complete.")
