import os
from dotenv import load_dotenv

# loaders
from langchain_community.document_loaders import (
    TextLoader,        # 👈 for text files
    PyPDFLoader,       # 👈 for PDFs
    WebBaseLoader      # 👈 for websites
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

api_key = str(os.getenv("OPENROUTER_API_KEY"))

# ─────────────────────────────────────────
# STEP 1 — Load ALL Sources
# ─────────────────────────────────────────
print("📚 Loading all sources...")
all_documents = []

# Load text files
print("\n📄 Loading text files...")
text_files = [
    "documents/sample.txt",
    "documents/notes.txt"
]
for file in text_files:
    try:
        loader = TextLoader(file, encoding="utf-8")
        docs = loader.load()
        all_documents.extend(docs)
        print(f"  ✅ Loaded: {file} ({len(docs)} doc)")
    except Exception as e:
        print(f"  ❌ Failed: {file} → {e}")

# Load PDFs
print("\n📑 Loading PDFs...")
pdf_files = [
    "documents/sample.pdf"
]
for file in pdf_files:
    try:
        loader = PyPDFLoader(file)
        docs = loader.load()
        all_documents.extend(docs)
        print(f"  ✅ Loaded: {file} ({len(docs)} pages)")
    except Exception as e:
        print(f"  ❌ Failed: {file} → {e}")

# Load websites
print("\n🌐 Loading websites...")
urls = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
]
try:
    loader = WebBaseLoader(urls)
    docs = loader.load()
    all_documents.extend(docs)
    print(f"  ✅ Loaded: {len(urls)} website(s) ({len(docs)} pages)")
except Exception as e:
    print(f"  ❌ Failed to load websites → {e}")

print(f"\n✅ Total documents loaded: {len(all_documents)}")

# ─────────────────────────────────────────
# STEP 2 — Split into Chunks
# ─────────────────────────────────────────
print("\n✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(all_documents)
print(f"✅ Created {len(chunks)} chunks total")

# show breakdown by source type
text_chunks = [c for c in chunks if c.metadata.get('source', '').endswith('.txt')]
pdf_chunks = [c for c in chunks if c.metadata.get('source', '').endswith('.pdf')]
web_chunks = [c for c in chunks if 'wikipedia' in c.metadata.get('source', '')]

print(f"  📄 Text chunks: {len(text_chunks)}")
print(f"  📑 PDF chunks:  {len(pdf_chunks)}")
print(f"  🌐 Web chunks:  {len(web_chunks)}")

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
    persist_directory="./chroma_db_advanced"
)
print("✅ Vector database created")

# ─────────────────────────────────────────
# STEP 5 — Create Retriever
# ─────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

# ─────────────────────────────────────────
# STEP 6 — Setup LLM
# ─────────────────────────────────────────
print("\n🤖 Setting up LLM...")
llm = ChatOpenAI(
    model="deepseek/deepseek-r1",
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0
)
print("✅ LLM ready")

# ─────────────────────────────────────────
# STEP 7 — Create Prompt Template
# ─────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant with access to 
     multiple knowledge sources including text files, PDFs 
     and websites. Answer based ONLY on the provided context.
     If the answer is not in the context, say 
     'I don't know based on the provided sources.'
     Always be specific and detailed in your answers.

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
# STEP 9 — Interactive Q&A with source info
# ─────────────────────────────────────────
print("\n" + "="*50)
print("🎯 ADVANCED RAG READY!")
print("Querying: Text files + PDFs + Websites")
print("Type 'exit' to quit")
print("="*50)

while True:
    question = input("\n❓ Your question: ")
    if question.lower() == "exit":
        print("Goodbye! 👋")
        break

    result = rag_chain.invoke({"input": question})
    print(f"\n💬 Answer: {result['answer']}")

    # show detailed source breakdown
    print("\n📚 Sources used:")
    for doc in result['context']:
        source = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page', '')

        # identify source type
        if source.endswith('.txt'):
            icon = "📄"
            source_type = "Text"
        elif source.endswith('.pdf'):
            icon = "📑"
            source_type = f"PDF (page {int(page)+1})" if page != '' else "PDF"
        elif 'http' in source:
            icon = "🌐"
            source_type = "Web"
        else:
            icon = "📁"
            source_type = "Unknown"

        print(f"  {icon} {source_type}: {source}")