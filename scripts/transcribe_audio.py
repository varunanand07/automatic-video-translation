import os
import json
import argparse
import yt_dlp
import whisper
import requests
import time

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

def download_audio(video_url, audio_dir):
    with yt_dlp.YoutubeDL({
        'format': 'bestaudio/best',
        'outtmpl': f'{audio_dir}/%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True
    }) as ydl:
        info = ydl.extract_info(video_url, download=True)
        video_id = info['id']
        audio_file = os.path.join(audio_dir, f"{video_id}.mp3")
    return video_id, audio_file

def transcribe_whisper(audio_file, transcript_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"Saved transcript to: {transcript_path}")

def transcribe_assemblyai(audio_file, transcript_path):
    headers = {
        "authorization": ASSEMBLYAI_API_KEY,
        "content-type": "application/json"
    }

    print("Uploading audio to AssemblyAI")
    with open(audio_file, 'rb') as f:
        response = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers={"authorization": ASSEMBLYAI_API_KEY},
            files={'file': f}
        )
    upload_url = response.json()["upload_url"]

    transcript_req = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        json={"audio_url": upload_url},
        headers=headers
    )
    transcript_id = transcript_req.json()["id"]

    polling_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    while True:
        poll_res = requests.get(polling_url, headers=headers).json()
        if poll_res["status"] == "completed":
            break
        elif poll_res["status"] == "error":
            raise Exception(f"Transcription failed: {poll_res['error']}")
        time.sleep(5)

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(poll_res["text"])
    print(f"Saved transcript to: {transcript_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ted_talks')
    parser.add_argument('--model', type=str, choices=['whisper', 'assemblyai'], default='whisper')
    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model
    base = f"data/{dataset}"
    audio_dir = f"{base}/audio"
    asr_dir = f"{base}/asr_transcripts/{model_name}"
    metadata_path = f"{base}/metadata.json"

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(asr_dir, exist_ok=True)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    video_urls = [entry["url"] for entry in metadata if "url" in entry]
    processed = 0

    for url in video_urls:
        try:
            print(f"\nProcessing: {url}")
            video_id, audio_file = download_audio(url, audio_dir)
            transcript_file = f"{asr_dir}/{video_id}.txt"
            if os.path.exists(transcript_file):
                print(f"This transcript already exists: {transcript_file}")
                continue
            if not os.path.exists(audio_file):
                print(f"Audio file is missing: {audio_file}")
                continue
            if model_name == "whisper":
                transcribe_whisper(audio_file, transcript_file)
            elif model_name == "assemblyai":
                transcribe_assemblyai(audio_file, transcript_file)

            processed += 1

        except Exception as e:
            print(f"Error processing {url}: {e}")
            continue

    print(f"\nTranscribed {processed}/{len(video_urls)} videos using {model_name}.")

if __name__ == '__main__':
    main()
