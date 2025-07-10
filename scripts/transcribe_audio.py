import whisper
import yt_dlp

video_urls = [
    "https://www.youtube.com/watch?v=-moW9jvvMr4&pp=ygUJdGVkIHRhbGtz",
    "https://www.youtube.com/watch?v=5MuIMqhT8DM&pp=ygUJdGVkIHRhbGtz0gcJCcEJAYcqIYzv",
    "https://www.youtube.com/watch?v=PY9DcIMGxMs&pp=ygUJdGVkIHRhbGtz"
]

model = whisper.load_model("base")

ydl_opts = {
    'format': 'bestaudio',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
    }]
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for url in video_urls:
        info = ydl.extract_info(url, download=True)
        video_id = info['id']
        audio_file = f"{video_id}.mp3" 
        result = model.transcribe(audio_file)
        print(result["text"]) 
