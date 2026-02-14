from dotenv import load_dotenv
import os
from openai import OpenAI
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import pyttsx3


load_dotenv()
client = OpenAI()

conversation = [
    {
        "role": "system",
        "content": "You are a friendly AI helping users improve their speech. Keep responses short and encouraging."
    }
]

whisper_model = whisper.load_model("small")

tts_engine = pyttsx3.init()

def record_audio(filename="input.wav", duration=5, fs=16000):
    print("🎤 Listening...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print("✅ Recording saved.")
    return filename


def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]


def speak(text):
    print("AI:", text)
    tts_engine.say(text)
    tts_engine.runAndWait()


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

    first_message = "Hi! I’d love to chat with you. What’s something you enjoy?"
    speak(first_message)

    while True:
        audio_file = record_audio(duration=5)
        user_text = transcribe_audio(audio_file)

        print("You said:", user_text)

        if user_text.lower() in ["exit", "quit"]:
            break

        ai_reply = get_ai_response(user_text)
        speak(ai_reply)
