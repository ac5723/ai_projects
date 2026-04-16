import ollama

# ─────────────────────────────────────────
# EXAMPLE 1 — Simple chat
# ─────────────────────────────────────────
print("="*50)
print("EXAMPLE 1: Simple Chat")
print("="*50)

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {"role": "user", "content": "What is Python programming?"}
    ]
)
print(f"Response: {response['message']['content']}")

# ─────────────────────────────────────────
# EXAMPLE 2 — Chat with system prompt
# ─────────────────────────────────────────
print("\n" + "="*50)
print("EXAMPLE 2: Chat with System Prompt")
print("="*50)

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful Python tutor. "
                      "Always give short, simple answers."
        },
        {
            "role": "user",
            "content": "What is a list in Python?"
        }
    ]
)
print(f"Response: {response['message']['content']}")

# ─────────────────────────────────────────
# EXAMPLE 3 — Streaming response
# ─────────────────────────────────────────
print("\n" + "="*50)
print("EXAMPLE 3: Streaming Response")
print("="*50)
print("Response: ", end="", flush=True)

stream = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {"role": "user", "content": "List 3 benefits of Python."}
    ],
    stream=True  # 👈 streams word by word like ChatGPT
)

for chunk in stream:
    print(chunk['message']['content'], end="", flush=True)
print()

# ─────────────────────────────────────────
# EXAMPLE 4 — Multi-turn conversation
# ─────────────────────────────────────────
print("\n" + "="*50)
print("EXAMPLE 4: Multi-turn Conversation")
print("="*50)

messages = []

questions = [
    "What is a variable in Python?",
    "Can you give me an example?",
    "What about a string variable?"
]

for question in questions:
    messages.append({"role": "user", "content": question})
    response = ollama.chat(
        model="llama3.2:3b",
        messages=messages
    )
    answer = response['message']['content']
    messages.append({"role": "assistant", "content": answer})
    print(f"\n❓ {question}")
    print(f"💬 {answer[:200]}...")  # first 200 chars

# ─────────────────────────────────────────
# EXAMPLE 5 — Generate embeddings
# ─────────────────────────────────────────
print("\n" + "="*50)
print("EXAMPLE 5: Generating Embeddings")
print("="*50)

texts = [
    "Python is a programming language",
    "Python is used for AI and ML",
    "Cricket is a popular sport in India"
]

for text in texts:
    embedding = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    vector = embedding['embedding']
    print(f"\nText: {text}")
    print(f"Embedding size: {len(vector)} dimensions")
    print(f"First 5 values: {vector[:5]}")