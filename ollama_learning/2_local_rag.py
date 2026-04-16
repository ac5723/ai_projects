import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# install missing package first if needed:
# pip install langchain-classic langchain-ollama

# ─────────────────────────────────────────
# Create sample document
# ─────────────────────────────────────────
os.makedirs("documents", exist_ok=True)
with open("documents/python_notes.txt", "w") as f:
    f.write("""
Python Learning Notes

Variables:
A variable stores data. Example: name = "Arpit", age = 25

Lists:
A list stores multiple items. Example: fruits = ["apple", "mango", "banana"]
Lists are mutable - you can change them after creation.

Dictionaries:
A dictionary stores key-value pairs.
Example: person = {"name": "Arpit", "age": 25, "city": "Mumbai"}

Functions:
A function is a reusable block of code.
def greet(name):
    return f"Hello {name}!"

Classes:
A class is a blueprint for creating objects.
class Dog:
    def __init__(self, name):
        self.name = name
    def bark(self):
        return f"{self.name} says Woof!"

Python Libraries for AI:
- NumPy: numerical computing
- Pandas: data manipulation
- Scikit-learn: machine learning
- TensorFlow: deep learning
- PyTorch: deep learning
- LangChain: LLM applications
- CrewAI: multi-agent systems
""")

# ─────────────────────────────────────────
# STEP 1 — Load Document
# ─────────────────────────────────────────
print("📄 Loading document...")
loader = TextLoader("documents/python_notes.txt", encoding="utf-8")
documents = loader.load()
print(f"✅ Loaded {len(documents)} document(s)")

# ─────────────────────────────────────────
# STEP 2 — Split into Chunks
# ─────────────────────────────────────────
print("\n✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

# ─────────────────────────────────────────
# STEP 3 — Local Embeddings with Ollama!
# ─────────────────────────────────────────
print("\n🔢 Creating LOCAL embeddings with Ollama...")
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"  # 👈 runs completely locally!
)
print("✅ Local embedding model ready")

# ─────────────────────────────────────────
# STEP 4 — Store in Vector Database
# ─────────────────────────────────────────
print("\n🗄️  Storing in vector database...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_local"
)
print("✅ Vector database created")

# ─────────────────────────────────────────
# STEP 5 — Create Retriever
# ─────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# ─────────────────────────────────────────
# STEP 6 — Local LLM with Ollama!
# ─────────────────────────────────────────
print("\n🤖 Setting up LOCAL LLM with Ollama...")
llm = OllamaLLM(
    model="llama3.2:3b",  # 👈 runs completely locally!
    temperature=0
)
print("✅ Local LLM ready")

# ─────────────────────────────────────────
# STEP 7 — Create Prompt
# ─────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful Python tutor.
     Answer ONLY based on the provided context.
     If answer is not in context, say 'Not in my notes.'

     Context: {context}"""),
    ("human", "{input}"),
])

# ─────────────────────────────────────────
# STEP 8 — Create RAG Chain
# ─────────────────────────────────────────
print("\n🔗 Creating LOCAL RAG chain...")
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)
print("✅ Fully LOCAL RAG ready — no internet needed!")

# ─────────────────────────────────────────
# STEP 9 — Ask Questions!
# ─────────────────────────────────────────
print("\n" + "="*50)
print("🎯 LOCAL RAG READY — 100% offline!")
print("Type 'exit' to quit")
print("="*50)

while True:
    question = input("\n❓ Your question: ")
    if question.lower() == "exit":
        print("Goodbye! 👋")
        break
    result = rag_chain.invoke({"input": question})
    print(f"\n💬 Answer: {result['answer']}")
    print(f"📚 Sources: {len(result['context'])} chunks")