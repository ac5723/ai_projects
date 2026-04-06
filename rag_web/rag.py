import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader  # 👈 web loader
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
# STEP 1 — Load Websites
# ─────────────────────────────────────────
print("🌐 Loading websites...")

urls = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Deep_learning",
]

loader = WebBaseLoader(urls)
documents = loader.load()
print(f"✅ Loaded {len(documents)} webpage(s)")

# print preview of each page
for i, doc in enumerate(documents):
    print(f"\n--- Page {i+1} Preview ---")
    print(doc.page_content[:200])  # first 200 chars
    print(f"Source: {doc.metadata.get('source', 'unknown')}")

# ─────────────────────────────────────────
# STEP 2 — Split into Chunks
# ─────────────────────────────────────────
print("\n✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

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
    persist_directory="./chroma_db_web"   # 👈 separate folder for web
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
    ("system", """You are a helpful assistant. Answer based ONLY 
     on the provided context from the websites.
     If the answer is not in the context, say 
     'I don't know based on the provided websites.'
     Always mention which topic your answer relates to.

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
# STEP 9 — Interactive Q&A
# ─────────────────────────────────────────
print("\n" + "="*50)
print("🎯 WEB RAG READY — Ask anything about the websites!")
print("Type 'exit' to quit")
print("="*50)

while True:
    question = input("\n❓ Your question: ")
    if question.lower() == "exit":
        print("Goodbye! 👋")
        break

    result = rag_chain.invoke({"input": question})
    print(f"\n💬 Answer: {result['answer']}")

    # show sources
    print("\n📚 Sources:")
    sources = set([doc.metadata.get('source', 'unknown')
                   for doc in result['context']])
    for source in sources:
        print(f"  → {source}")