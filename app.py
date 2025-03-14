from flask import Flask, render_template, request, jsonify
import os 
import sounddevice as sd
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel

app = Flask(__name__)

# Initialize the Whisper model
model_size = "medium.en"
model = WhisperModel(model_size, device="cpu", compute_type="float32")

# Function to record audio chunk
def record_chunk(file_path, duration=10, fs=16000):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    sf.write(file_path, recording, fs)

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
