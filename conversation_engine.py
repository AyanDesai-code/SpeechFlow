CONVERSATION_LIMIT = 120  # seconds

from dotenv import load_dotenv
import os
import time
import uuid
import re
import numpy as np
import pandas as pd
from collections import Counter

from openai import OpenAI
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import pyttsx3

import main



load_dotenv()
client = OpenAI()

conversation = [
    {
        "role": "system",
        "content": "You are a friendly conversational AI. Keep responses short and concise. Only speak English. No emojis."
    }
]

print("Loading Whisper model...")
whisper_model = whisper.load_model("small")
print("Whisper loaded.")

tts_engine = pyttsx3.init()
audio_files = []


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


def transcribe(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]


def speak(text):
    print("AI:", text)
    tts_engine.say(text)
    tts_engine.runAndWait()
    time.sleep(0.6)


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


def clean_words(text):
    return re.findall(r"[a-zA-Z]+", text.lower())


def force_practice(word):

    word = word.lower()
    speak(f"You must say the word {word}. Say it once clearly.")

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
            speak("Good. That was clear.")
            break
        else:
            speak("Not clear. Say it once slowly.")


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

    repetition_counter = Counter()

    for file in audio_files:

        print("Processing", file)

        try:
            _, text_df = main.process_audio(file)
        except Exception as e:
            print("Skipping", file, e)
            continue

        raw_words = text_df["text"].fillna("").tolist()
        words = [w.lower() for w in raw_words if w]

        raw_text = " ".join(words)

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


    FILLER_WORDS = {"uh", "um", "erm", "ah", "uhh", "umm"}

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
    speak("Hi. What is something you enjoy?")

    while True:

        if time.time() - start_time > CONVERSATION_LIMIT:
            speak("That was a good conversation. Let us stop here.")
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

    print("\nAnalyzing session...\n")

    practice_words = analyze_session(audio_files)

    if not practice_words:
        speak("You spoke smoothly. Good job.")
    else:
        top_words = [w for w, _ in practice_words.most_common(5)]
        speak("We will practice these words: " + ", ".join(top_words))

        for word in top_words:
            force_practice(word)

        speak("Session complete. Excellent work.")
