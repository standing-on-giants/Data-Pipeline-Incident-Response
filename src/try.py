from groq import Groq

client = Groq(api_key="")

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # or mixtral-8x7b-32768, gemma2-9b-it
    messages=[
        {"role": "user", "content": "Hello! What can you do?"}
    ]
)

print(response.choices[0].message.content) 
