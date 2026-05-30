CONVERSATION_LIMIT = 150  # seconds
import torchaudio
import os
from silero_vad import load_silero_vad, get_speech_timestamps
import time
import uuid
import re
import numpy as np
import pandas as pd
from collections import Counter
from dotenv import load_dotenv
import threading
from openai import OpenAI
import sounddevice as sd
from scipy.io.wavfile import write
import whisper_timestamped as whisper
import pyttsx3
import main
import requests
import win32com.client
from playsound import playsound
import edge_tts
import asyncio
import torch
import base64
import json

import transformers
tts_lock = threading.Lock()
model=load_silero_vad()
torch.set_num_threads(4)
def log(text, speak_text=False):

    text = str(text)


    try:
        requests.post(
            "http://192.168.1.185:5000/log",
            json={"message": text},
            timeout=1
        )
    except Exception as e:
        print("Log failed:", e)

    if speak_text:
            try:
                with tts_lock:
                    communicate = edge_tts.Communicate(text, "en-US-JennyNeural", rate="+30%")
                    asyncio.run(communicate.save("temp_response.mp3"))
                    playsound("temp_response.mp3")
            except Exception as e:
                print("TTS failed:", e)


load_dotenv()
client = OpenAI(
    base_url=os.getenv("base_url"),
    api_key=os.getenv("api_key")
)


conversation = [
    {
        "role": "system",
        "content": "You are a friendly conversational AI. Keep responses short and concise. Only speak English. No emojis."
    }
]
log("Loading the model, speak when the text says: 'Listening for speech'", speak_text=True)
print("Loading Whisper model...")
device='cpu'
whisper_model = whisper.load_model('demo_models/asr', device='cpu')
whisper_model.to(device)
audio_files = []


def record_audio(
    fs=16000,
    frame_ms=30,
    silence_duration=1.5,
    max_duration=20
):
    import numpy as np
    import torch
    import time
    import uuid
    from scipy.io.wavfile import write

    frame_size = 512 #int(fs * frame_ms / 1000)

    recording = []
    speech_started = False

    silence_start = None
    start_time = time.time()

    filename = f"user_{uuid.uuid4().hex[:8]}.wav"

    log("Listening for speech...")

    with sd.InputStream(samplerate=fs, channels=1, dtype="float32") as stream:

        while True:
            data, _ = stream.read(frame_size)
            audio_chunk = data.flatten()

            # convert to tensor for silero
            audio_tensor = torch.from_numpy(audio_chunk.copy())

            # Silero VAD prediction (batch-sized but fast per frame)
            speech_prob = model(audio_tensor, fs).item()
            print(f"Speech probability: {speech_prob:.2f}")
            is_speech = speech_prob > 0.95

            # ----------------------------
            # 1. Speech START detection
            # ----------------------------
            if not speech_started:
                if is_speech:
                    speech_started = True
                    log("Speech detected → starting recording")
                    recording.append(audio_chunk)
                    start_time = time.time()
                continue

            # ----------------------------
            # 2. Already recording
            # ----------------------------
            recording.append(audio_chunk)

            # max duration safety
            if time.time() - start_time > max_duration:
                log("Max duration reached")
                break

            # ----------------------------
            # 3. Silence detection (for stop)
            # ----------------------------
            if is_speech:
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > silence_duration:
                    log("Silence detected → stopping recording")
                    break

    # ----------------------------
    # No speech recorded fallback
    # ----------------------------
    if not speech_started:
        return None

    audio = np.concatenate(recording, axis=0).flatten()

    # convert to int16 WAV
    audio_int16 = (audio * 32767).astype(np.int16)

    write(filename, fs, audio_int16)
    print("Saved:", filename)

    return filename

def transcribe(audio_path):
    log("Transcribing...")


    with open(audio_path, "rb") as f:
        base64_audio = base64.b64encode(f.read()).decode("utf-8")

    response = requests.post(
    url="https://openrouter.ai/api/v1/audio/transcriptions",
    headers={
        "Authorization": f"Bearer {os.getenv('api_key')}",
        "Content-Type": "application/json"
        
    },
    data=json.dumps({
        "model": "openai/gpt-4o-mini-transcribe",
        "input_audio": {
        "data": base64_audio,
        "format": "wav"
        },
        "timestamp_granularities": ["segment"]

    })
    )
    result = response.json()
    print(result)
    return result["text"]







def get_ai_response(user_text):
    log("Getting AI response...")
    conversation.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation,
        temperature=0.6
    )

    ai_text = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": ai_text})
    return ai_text


def clean_words(text):
    return re.findall(r"[a-zA-Z]+", text.lower())


def force_practice(word):

    word = word.lower()
    log(f"You must say the word {word}. Say it once clearly.", speak_text=True)

    while True:

        audio_file = record_audio()
        if audio_file is None:
            continue

        transcript = transcribe(audio_file)
        print("Heard:", transcript)

        words = clean_words(transcript)

        correct = word in words
        short_enough = len(words) <= 2

        repetition = False
        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                repetition = True

        if correct and short_enough and not repetition:
            log("Good. That was clear.", speak_text=True)
            break
        else:
            log("Not clear. Say it once slowly.", speak_text=True)


def has_prefix_stutter(word):
    word = word.lower().replace("-", "")
    if len(word) < 6:
        return False

    for size in [1, 2, 3]:
        prefix = word[:size]
        if word.startswith(prefix * 3):
            return True

    return False


def has_hyphen_stutter(word):
    parts = word.split("-")
    if len(parts) >= 3 and all(p == parts[0] for p in parts[:-1]):
        return True
    return False


def analyze_session(audio_files):
    log("Analyzing session...")
    repetition_counter = Counter()

    for file in audio_files:

        log(f"Processing {file}")
        file_id = file.replace(".wav", "")
        try:
            _, text_df = main.process_audio(file, modality="multimodal", output_trans=f"{file_id}output.csv", output_file=f"{file_id}.csv")
        except Exception as e:
            print("Skipping", file, e)
            continue

        # Check disfluency CSV for this file
        disfluency_file = f"{file_id}.csv"
        if os.path.exists(disfluency_file):
            try:
                disfluency_df = pd.read_csv(disfluency_file)
                disfluency_columns = ["FP", "RP", "RV", "RS", "PW"]
                disfluency_names = {
                    "FP": "Filled Pause",
                    "RP": "Repetition",
                    "RV": "Revision",
                    "RS": "Restart",
                    "PW": "Partial Word"
                }
                
                for col in disfluency_columns:
                    if col in disfluency_df.columns:
                        disfluencies = disfluency_df[disfluency_df[col] == 1]
                        if len(disfluencies) > 0:
                            start_time = disfluencies.iloc[0]["frame_time"]
                            end_time = disfluencies.iloc[-1]["frame_time"]
                            count = len(disfluencies)
                            log(f"  {disfluency_names[col]}: {count} frames detected from {start_time:.2f}s to {end_time:.2f}s")
                        else:
                            log(f"  {disfluency_names[col]}: No frames detected")
            except Exception as e:
                print(f"Error reading disfluency CSV {disfluency_file}: {e}")

        raw_words = text_df["text"].fillna("").tolist()
        words = [w.lower() for w in raw_words if w]

        raw_text = " ".join(words)
        print("Raw text:", raw_text)
        matches = re.findall(r"\b(\w+)( \1){2,}", raw_text)
        for match in matches:
            repetition_counter[match[0]] += 3

        for word in words:
            if has_prefix_stutter(word):
                repetition_counter[word] += 3
            if has_hyphen_stutter(word):
                repetition_counter[word] += 3

        i = 0
        while i < len(words):
            run_length = 1
            while i + run_length < len(words) and words[i] == words[i + run_length]:
                run_length += 1

            if run_length >= 2:
                repetition_counter[words[i]] += run_length

            i += run_length

        for i in range(len(words) - 3):
            if words[i:i+2] == words[i+2:i+4]:
                phrase = " ".join(words[i:i+2])
                repetition_counter[phrase] += 2



    FILLER_WORDS = {"uh", "um", "erm", "ah", "uhh", "umm", "like", "you know", "i mean", "so", "actually", "basically", "right", "well", "hmm"}

    def valid(word):
        return (
            word not in FILLER_WORDS and
            word.replace("-", "").isalpha() and
            len(word) >= 4
        )

    practice_words = Counter()

    for word, count in repetition_counter.items():
        if count >= 2 and valid(word):
            practice_words[word] += count

    return practice_words



if __name__ == "__main__":

    start_time = time.time()
    log("Hi. What is something you enjoy?", speak_text=True)

    while True:

        if time.time() - start_time > CONVERSATION_LIMIT:
            log("That was a good conversation. Let us stop here.", speak_text=True)
            break

        audio_file = record_audio()
        if audio_file is None:
            continue

        audio_files.append(audio_file)

        
        user_text = transcribe(audio_file)
        print("You said:", user_text)
        log(f"You said: {user_text}", speak_text=False)

        if user_text.strip().lower() in ["exit", "quit", "Exit", "Quit", "exit.", "Exit."]:
            break

        if not user_text.strip():
            continue
        ai_reply = get_ai_response(user_text)
        print("Hey we got here")
        log(f"{ai_reply}", speak_text=True)

    print("\nAnalyzing session...\n")

    practice_words = analyze_session(audio_files)
    #log(list(practice_words.keys()))

    if not practice_words:
        log("There are no specific words you need to practice. Good job.", speak_text=True)
    else:
        top_words = [w for w, _ in practice_words.most_common(5)]
        log(f"Words we noticed you could improve on: {', '.join(top_words)}", speak_text=True)

        for word in top_words:
            force_practice(word)

        log("Session complete. Excellent work.", speak_text=True)
