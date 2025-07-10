import whisper
import yt_dlp
import os

video_urls = [
    "https://www.youtube.com/watch?v=-moW9jvvMr4&pp=ygUJdGVkIHRhbGtz",
    "https://www.youtube.com/watch?v=5MuIMqhT8DM&pp=ygUJdGVkIHRhbGtz0gcJCcEJAYcqIYzv",
    "https://www.youtube.com/watch?v=PY9DcIMGxMs&pp=ygUJdGVkIHRhbGtz"
]

model = whisper.load_model("base")

audio_dir = "data/audio"
transcript_dir = "data/asr_transcripts"

ydl_opts = {
    'format': 'bestaudio',
    'outtmpl': f'{audio_dir}/%(id)s.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
    }]
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for url in video_urls:
        info = ydl.extract_info(url, download=True)
        video_id = info['id']
        audio_path = f"{audio_dir}/{video_id}.mp3"
        result = model.transcribe(audio_path)

        with open(f"{transcript_dir}/{video_id}.txt", "w", encoding="utf-8") as f:
            f.write(result["text"])
