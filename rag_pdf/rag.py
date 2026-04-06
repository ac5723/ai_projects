import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader  # 👈 changed!
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
# STEP 1 — Load PDF
# ─────────────────────────────────────────
print("📄 Loading PDF...")
loader = PyPDFLoader("documents/sample.pdf")  # 👈 changed!
documents = loader.load()
print(f"✅ Loaded {len(documents)} page(s)")

# ─────────────────────────────────────────
# STEP 2 — Split into Chunks
# ─────────────────────────────────────────
print("\n✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 👈 bigger chunks for PDFs
    chunk_overlap=100    # 👈 more overlap for PDFs
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
    persist_directory="./chroma_db_pdf"   # 👈 different folder for PDF
)
print("✅ Vector database created")

# ─────────────────────────────────────────
# STEP 5 — Create Retriever
# ─────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# ─────────────────────────────────────────
# STEP 6 — Setup LLM
# ─────────────────────────────────────────
print("\n🤖 Setting up LLM...")
llm = ChatOpenAI(
    model="anthropic/claude-sonnet-4-6",
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0
)
print("✅ LLM ready")

# ─────────────────────────────────────────
# STEP 7 — Create Prompt Template
# ─────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a precise assistant. Answer ONLY based on 
     the provided context. Be specific and detailed.
     If the answer is not clearly stated in the context, 
     say 'I don't know based on the provided documents.'
     Do not make assumptions or add information not in the context.

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
# STEP 9 — Interactive Q&A loop!
# ─────────────────────────────────────────
print("\n" + "="*50)
print("🎯 PDF RAG READY — Ask anything about your PDF!")
print("Type 'exit' to quit")
print("="*50)

while True:
    question = input("\n❓ Your question: ")
    if question.lower() == "exit":
        print("Goodbye! 👋")
        break
    result = rag_chain.invoke({"input": question})
    print(f"\n💬 Answer: {result['answer']}")
    print(f"📚 Found in {len(result['context'])} chunks")
    print(f"📄 Pages: {set([doc.metadata.get('page', '?') + 1 for doc in result['context']])}")