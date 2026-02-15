CONVERSATION_LIMIT = 120  # seconds

from dotenv import load_dotenv
import os
from openai import OpenAI
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import pyttsx3
import numpy as np
import time
import uuid
import main
print("Imported main from:", main.__file__)


load_dotenv()
client = OpenAI()

conversation = [
    {
        "role": "system",
        "content": "You are a friendly conversational ai that is engaging and addictive and likes to chat about anything the user likes"
    }
]

print("Loading Whisper model...")
whisper_model = whisper.load_model("small")
print("Whisper loaded.")

tts_engine = pyttsx3.init()

def record_audio(fs=16000,
                 silence_threshold=0.002,
                 silence_duration=2.5,
                 max_duration=20):

    recording = []
    silence_start = None
    start_time = None
    recording_started = False

    filename = f"user_{uuid.uuid4().hex[:8]}.wav"

    with sd.InputStream(samplerate=fs,
                        channels=1,
                        dtype='float32') as stream:

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

    if recording_started:
        audio = np.concatenate(recording, axis=0)
        audio = audio.flatten()

        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        write(filename, fs, audio_int16)

        print(f"Saved: {filename}")
        return filename
    else:
        print("⚠ No speech detected.")
        return None

def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]



def speak(text):
    print("AI:", text)
    tts_engine.say(text)
    tts_engine.runAndWait()
    time.sleep(1.5)



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

if __name__ == "__main__":

    start_time = time.time()

    first_message = "Hi! I’d love to chat with you. What’s something you enjoy?"
    speak(first_message)

    while True:

        # ⏰ Timer Check
        if time.time() - start_time > CONVERSATION_LIMIT:
            speak("That was a great conversation! Let's stop here for now.")
            print("⏰ Conversation time limit reached.")
            break

        time.sleep(1.0)

        audio_file = record_audio()

        if audio_file is None:
            continue

        # 🔥 SEND AUDIO TO main.py IMMEDIATELY
        print("Running multimodal disfluency analysis...")
        predictions = main.process_audio(audio_file)
        print("Model Output Preview:")
        print(predictions.head())

        # Continue conversation
        user_text = transcribe_audio(audio_file)
        print("You said:", user_text)

        if user_text.strip().lower() in ["exit", "quit"]:
            break

        if user_text.strip() == "":
            print("⚠ Empty transcript.")
            continue

        ai_reply = get_ai_response(user_text)
        speak(ai_reply)
