import whisper
import yt_dlp
import os
import json

audio_dir = "data/audio"
asr_dir = "data/asr_transcripts"
metadata_path = "data/metadata.json"
os.makedirs(audio_dir, exist_ok=True)
os.makedirs(asr_dir, exist_ok=True)

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

video_urls = [entry["url"] for entry in metadata if "url" in entry]

model = whisper.load_model("base")

ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': f'{audio_dir}/%(id)s.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'quiet': True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for url in video_urls:
        try:
            print(f"\n Downloading and transcribing the videos from: {url}")
            info = ydl.extract_info(url, download=True)
            video_id = info['id']
            audio_file = f"{audio_dir}/{video_id}.mp3"
            transcript_file = f"{asr_dir}/{video_id}.txt"

            if os.path.exists(transcript_file):
                print(f"There is already a transcript for this video: {transcript_file}")
                continue

            result = model.transcribe(audio_file)
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(result["text"])
            print(f"Saved automatically generated transcript: {transcript_file}")

        except Exception as e:
            print(f"Error on {url}: {e}")
