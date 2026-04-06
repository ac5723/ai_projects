import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain                          # ✅
from langchain_classic.chains.combine_documents import create_stuff_documents_chain  # ✅
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


api_key = str(os.getenv("OPENROUTER_API_KEY"))
print(f"API Key loaded: {api_key[:10]}...")  # prints first 10 chars only

# ─────────────────────────────────────────
# STEP 1 — Load Document
# ─────────────────────────────────────────
print("📄 Loading document...")
loader = TextLoader("documents/sample.txt", encoding="utf-8")
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

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)

# ─────────────────────────────────────────
# STEP 3 — Create Embeddings
# ─────────────────────────────────────────
print("\n🔢 Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("✅ Embedding model loaded")

# ─────────────────────────────────────────
# STEP 4 — Store in Vector Database
# ─────────────────────────────────────────
print("\n🗄️  Storing in vector database...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("✅ Vector database created")

# ─────────────────────────────────────────
# STEP 5 — Create Retriever
# ─────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# ─────────────────────────────────────────
# STEP 6 — Setup LLM via OpenRouter
# ─────────────────────────────────────────
print("\n🤖 Setting up LLM...")

api_key = str(os.getenv("OPENROUTER_API_KEY"))

llm = ChatOpenAI(
    model="openrouter/auto",       # 👈 correct OpenRouter model
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0
)
print("✅ LLM ready")

# ─────────────────────────────────────────
# STEP 7 — Create Prompt Template
# ─────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Use the following 
     context to answer the question. If the answer is not in 
     the context, say 'I don't know based on the provided documents.'

     Context: {context}"""),
    ("human", "{input}"),
])

# ─────────────────────────────────────────
# STEP 8 — Create RAG Chain
# ─────────────────────────────────────────
print("\n🔗 Creating RAG chain...")
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)
print("✅ RAG chain ready")

# ─────────────────────────────────────────
# STEP 9 — Ask Questions!
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("🎯 RAG SYSTEM READY — Ask your questions!")
print("=" * 50)

questions = [
    "What is the refund policy?",
    "How much does CloudSync cost?",
    "Who is the CEO?",
    "What are the support hours?",
    "When was AIAssist launched?",
    "What is the capital of France?"
]

for question in questions:
    print(f"\n❓ Question: {question}")
    result = rag_chain.invoke({"input": question})
    print(f"💬 Answer: {result['answer']}")
    print(f"📚 Sources used: {len(result['context'])} chunks")