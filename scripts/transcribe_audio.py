import whisper
import yt_dlp
import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ted_talks')
    args = parser.parse_args()

    base = f"data/{args.dataset}"
    audio_dir = f"{base}/audio"
    asr_dir = f"{base}/asr_transcripts"
    original_dir = f"{base}/original_transcripts"
    metadata_path = f"{base}/metadata.json"
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

    processed = 0
    for url in video_urls:
        try:
            print(f"\nDownloading and transcribing: {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_id = info['id']
                audio_file = f"{audio_dir}/{video_id}.mp3"
                transcript_file = f"{asr_dir}/{video_id}.txt"
                if os.path.exists(transcript_file):
                    print(f"Transcript already exists: {transcript_file}")
                    continue
                result = model.transcribe(audio_file)
                with open(transcript_file, "w", encoding="utf-8") as f:
                    f.write(result["text"])
                print(f"Saved transcript: {transcript_file}")
                processed += 1
        except Exception as e:
            print(f"Error on {url}: {e}")
            continue
    print(f"\nTranscribed {processed}/{len(video_urls)} videos.")

if __name__ == '__main__':
    main()
