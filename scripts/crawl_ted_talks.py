import yt_dlp
import os
import json
import webvtt
import time
import random

transcript_dir = "data/original_transcripts"
reference_dir = "data/reference_translations"
metadata_file = "data/metadata.json"
os.makedirs(transcript_dir, exist_ok=True)
os.makedirs(reference_dir, exist_ok=True)

video_urls = [
    "https://www.youtube.com/watch?v=-moW9jvvMr4&pp=ygUJdGVkIHRhbGtz",
    "https://www.youtube.com/watch?v=5MuIMqhT8DM&pp=ygUJdGVkIHRhbGtz0gcJCcEJAYcqIYzv",
    "https://www.youtube.com/watch?v=PY9DcIMGxMs&pp=ygUJdGVkIHRhbGtz",
    "https://www.youtube.com/watch?v=BHY0FxzoKZE",
    "https://www.youtube.com/watch?v=6Af6b_wyiwI",
    "https://www.youtube.com/watch?v=P6FORpg0KVo",
    "https://www.youtube.com/watch?v=4TMPXK9tw5U",
    "https://www.youtube.com/watch?v=LBvHI1awWaI",
    "https://www.youtube.com/watch?v=LnJwH_PZXnM",
    "https://www.youtube.com/watch?v=MdZAMSyn_As"
]

metadata = []

ydl_opts = {
    'skip_download': True,
    'writesubtitles': True,
    'writeautomaticsub': True,
    'subtitleslangs': ['en', 'es'],
    'subtitlesformat': 'vtt',
    'outtmpl': 'data/original_transcripts/%(id)s.%(ext)s',
    'quiet': True,
    'retries': 3,
    'fragment_retries': 3,
    'extractor_retries': 3,
    'http_chunk_size': 10485760,  
    'sleep_interval': 2,
    'max_sleep_interval': 10,
    'sleep_interval_requests': 1,
    'max_sleep_interval_requests': 5
}

def vtt_to_txt(vtt_path, txt_path):
    try:
        text_lines = []
        for caption in webvtt.read(vtt_path):
            text_lines.append(caption.text.strip())
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_lines))
    except Exception as e:
        print(f"Error processing {vtt_path}: {e}")

def download_with_retry(ydl, url, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"Attempting to download {url} (attempt {attempt + 1}/{max_retries})")
            info = ydl.extract_info(url, download=True)
            return info
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait_time = (2 ** attempt) + random.uniform(1, 5)
                print(f"The script has been rate limited. Waiting {wait_time:.1f} seconds before retry.")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    raise e
            else:
                raise e
    return None

for i, url in enumerate(video_urls):
    try:
        print(f"\nProcessing video {i+1}/{len(video_urls)}: {url}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = download_with_retry(ydl, url)
            if not info:
                print(f"Failed to download info for {url}")
                continue
                
            video_id = info['id']
            print(f"Video ID: {video_id}")

            vtt_file_en = f"data/original_transcripts/{video_id}.en.vtt"
            txt_file_en = f"data/original_transcripts/{video_id}.txt"
            if os.path.exists(vtt_file_en):
                print(f"Converting English VTT to TXT for {video_id}")
                vtt_to_txt(vtt_file_en, txt_file_en)
                os.remove(vtt_file_en)
            else:
                print(f"No English captions found for {video_id}")

            vtt_file_es = f"data/original_transcripts/{video_id}.es.vtt"
            txt_file_es = f"data/reference_translations/{video_id}_es.txt"
            if os.path.exists(vtt_file_es):
                print(f"Converting Spanish VTT to TXT for {video_id}")
                vtt_to_txt(vtt_file_es, txt_file_es)
                os.remove(vtt_file_es)
            else:
                print(f"No Spanish captions found for {video_id}")

            metadata.append({
                'id': video_id,
                'title': info.get('title'),
                'url': url,
                'duration': info.get('duration'),
                'subtitles_available': os.path.exists(txt_file_en)
            })

        if i < len(video_urls) - 1:
            delay = random.uniform(3, 7)
            print(f"Waiting {delay:.1f} seconds before processing the next video")
            time.sleep(delay)

    except Exception as e:
        print(f"Failed to process {url}: {e}")
        continue

print(f"\nSaving metadata for {len(metadata)} videos.")
with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print("Crawling for TED talks completed.")
