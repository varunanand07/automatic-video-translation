import yt_dlp
import os
import json
import webvtt

transcript_dir = "data/original_transcripts"
metadata_file = "data/metadata.json"
os.makedirs(transcript_dir, exist_ok=True)

video_urls = [
    "https://www.youtube.com/watch?v=-moW9jvvMr4&pp=ygUJdGVkIHRhbGtz",
    "https://www.youtube.com/watch?v=5MuIMqhT8DM&pp=ygUJdGVkIHRhbGtz0gcJCcEJAYcqIYzv",
    "https://www.youtube.com/watch?v=PY9DcIMGxMs&pp=ygUJdGVkIHRhbGtz"
]

metadata = []

ydl_opts = {
    'skip_download': True,
    'writesubtitles': True,
    'writeautomaticsub': True,
    'subtitleslangs': ['en'],
    'subtitlesformat': 'vtt',
    'outtmpl': 'data/original_transcripts/%(id)s.%(ext)s',
    'quiet': True
}

def vtt_to_txt(vtt_path, txt_path):
    try:
        text_lines = []
        for caption in webvtt.read(vtt_path):
            text_lines.append(caption.text.strip())
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_lines))

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for url in video_urls:
        try:
            info = ydl.extract_info(url, download=True)
            video_id = info['id']
            vtt_file = f"data/original_transcripts/{video_id}.en.vtt"
            txt_file = f"data/original_transcripts/{video_id}.txt"
            if os.path.exists(vtt_file):
                vtt_to_txt(vtt_file, txt_file)
                os.remove(vtt_file)
            else:
                print(f"No captions found for {url}")
            
            metadata.append({
                'id': video_id,
                'title': info.get('title'),
                'url': url,
                'duration': info.get('duration'),
                'subtitles_available': os.path.exists(txt_file)
            })

with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
