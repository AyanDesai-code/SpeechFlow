from dotenv import load_dotenv
import os
from openai import OpenAI

client = OpenAI()

conversation = [
    {"role": "system", "content": "You are a friendly AI that helps users improve their speech."}
]

while True:
    user_input = input("User: ")
    
    conversation.append({"role": "user", "content": user_input})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation
    )
    
    ai_reply = response.choices[0].message.content
    print("AI:", ai_reply)
    
    conversation.append({"role": "assistant", "content": ai_reply})

