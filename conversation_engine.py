CONVERSATION_LIMIT = 120  # seconds

from dotenv import load_dotenv
import os
import time
import uuid
import numpy as np
import pandas as pd
from collections import Counter

from openai import OpenAI
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import pyttsx3

import main  # your rebuilt main.py


# ==========================================
# SETUP
# ==========================================

load_dotenv()
client = OpenAI()

conversation = [
    {
        "role": "system",
        "content": "You are a friendly conversational AI that is engaging and likes to chat about anything the user enjoys."
    }
]

print("Loading Whisper model...")
whisper_model = whisper.load_model("small")
print("Whisper loaded.")

tts_engine = pyttsx3.init()

audio_files = []


# ==========================================
# RECORD AUDIO
# ==========================================

def record_audio(fs=16000,
                 silence_threshold=0.002,
                 silence_duration=2.0,
                 max_duration=20):

    recording = []
    silence_start = None
    recording_started = False
    start_time = None

    filename = f"user_{uuid.uuid4().hex[:8]}.wav"

    with sd.InputStream(samplerate=fs,
                        channels=1,
                        dtype="float32") as stream:

        while True:
            data, _ = stream.read(1024)
            volume = np.linalg.norm(data) / len(data)

            if not recording_started:
                if volume > silence_threshold:
                    recording_started = True
                    start_time = time.time()
                    recording.append(data)
                continue

            recording.append(data)

            if volume < silence_threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > silence_duration:
                    break
            else:
                silence_start = None

            if time.time() - start_time > max_duration:
                break

    if not recording_started:
        return None

    audio = np.concatenate(recording, axis=0).flatten()
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    write(filename, fs, audio_int16)
    print("Saved:", filename)

    return filename


# ==========================================
# SPEECH / AI
# ==========================================

def transcribe(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]


def speak(text):
    print("AI:", text)
    tts_engine.say(text)
    tts_engine.runAndWait()
    time.sleep(0.8)


def get_ai_response(user_text):
    conversation.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation,
        temperature=0.6
    )

    ai_text = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": ai_text})
    return ai_text


# ==========================================
# PRACTICE MODE
# ==========================================

def force_practice(word):

    speak(f"You must say the word: {word}. Say it once, clearly.")

    while True:

        audio_file = record_audio()

        if audio_file is None:
            continue

        transcript = transcribe(audio_file).lower().strip()
        words = transcript.split()

        try:
            pred_df, text_df = main.process_audio(audio_file)
        except Exception:
            speak("Try again.")
            continue

        if len(pred_df) > 0:
            fp_ratio = pred_df["FP"].mean()
        else:
            fp_ratio = 0

        repetition = any(
            words[i] == words[i - 1]
            for i in range(1, len(words))
        )

        correct = word.lower() in words
        single = len(words) <= 2

        if correct and not repetition and fp_ratio < 0.6 and single:
            speak("Good. That was clear.")
            break
        else:
            speak("Not clear. Say it once, slowly.")


# ==========================================
# SESSION ANALYSIS
# ==========================================

def analyze_session(audio_files):

    fp_word_counter = Counter()
    repetition_counter = Counter()

    for file in audio_files:

        print("Processing", file)

        try:
            pred_df, text_df = main.process_audio(file)
        except Exception as e:
            print("Skipping", file, e)
            continue

        words = text_df["text"].fillna("").str.lower().tolist()

        # repetition detection
        for i in range(1, len(words)):
            if words[i] == words[i - 1] and words[i] != "":
                repetition_counter[words[i]] += 1

        # FP mapping
        for _, row in text_df.iterrows():

            if pd.isna(row["start"]) or pd.isna(row["end"]):
                continue

            start_frame = int(row["start"] / 0.02)
            end_frame = int(row["end"] / 0.02)
            end_frame = min(end_frame, len(pred_df) - 1)

            if end_frame <= start_frame:
                continue

            word_frames = pred_df.iloc[start_frame:end_frame]

            if len(word_frames) == 0:
                continue

            if word_frames["FP"].mean() > 0.5:
                fp_word_counter[row["text"].lower()] += 1

    # ======================================
    # CLEAN WORD FILTERING
    # ======================================

    FILLER_WORDS = {"uh", "um", "erm", "ah", "uhh", "umm"}

    def is_valid_practice_word(word):
        return (
            word not in FILLER_WORDS and
            word.isalpha() and
            len(word) >= 4
        )

    practice_words = Counter()

    for word, count in fp_word_counter.items():
        if count > 1 and is_valid_practice_word(word):
            practice_words[word] += count

    for word, count in repetition_counter.items():
        if count > 1 and is_valid_practice_word(word):
            practice_words[word] += count

    return practice_words


# ==========================================
# MAIN LOOP
# ==========================================

if __name__ == "__main__":

    start_time = time.time()

    speak("Hi! What’s something you enjoy?")

    while True:

        if time.time() - start_time > CONVERSATION_LIMIT:
            speak("That was a great conversation! Let's stop here.")
            break

        audio_file = record_audio()

        if audio_file is None:
            continue

        audio_files.append(audio_file)

        user_text = transcribe(audio_file)
        print("You said:", user_text)

        if user_text.strip().lower() in ["exit", "quit"]:
            break

        if not user_text.strip():
            continue

        ai_reply = get_ai_response(user_text)
        speak(ai_reply)

    # ======================================
    # AFTER SESSION
    # ======================================

    print("\nAnalyzing session...\n")

    practice_words = analyze_session(audio_files)

    if not practice_words:
        speak("You spoke smoothly. Great job.")
    else:
        top_words = [w for w, _ in practice_words.most_common(5)]

        speak(
            "We will practice these words: "
            + ", ".join(top_words)
        )

        for word in top_words:
            force_practice(word)

        speak("Session complete. Excellent work.")
