from flask import Flask, render_template, request, jsonify
import os
import pyaudio
import wave
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel

app = Flask(__name__)

# Initialize the Whisper model
model_size = "medium.en"
model = WhisperModel(model_size, device="cpu", compute_type="float32")

# Function to record audio chunk
def record_chunk(file_path, duration=10, fs=16000):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)
    frames = []

    for _ in range(0, int(fs / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))

# Function to transcribe audio chunk
def transcribe_chunk(model, file_path):
    audio, _ = sf.read(file_path)
    segments_generator, _ = model.transcribe(audio)
    transcription = ''.join(segment.text for segment in segments_generator)
    return transcription

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    chunk_file = "temp_chunk.wav"
    record_chunk(chunk_file, duration=10)
    transcription = transcribe_chunk(model, chunk_file)
    os.remove(chunk_file)
    return jsonify({'transcription': transcription})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
