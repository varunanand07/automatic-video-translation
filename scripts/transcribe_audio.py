import whisper

model = whisper.load_model("base")

audio_path = "Ks-_Mh1QhMc.mp3"

result = model.transcribe(audio_path)
print(result["text"])
