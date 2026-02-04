from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Eres un asistente jurídico."},
        {"role": "user", "content": "Explica brevemente qué es la ratio decidendi."}
    ],
    temperature=0.2
)

print(resp.choices[0].message.content)

