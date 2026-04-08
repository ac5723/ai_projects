import os
from dotenv import load_dotenv
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# ─────────────────────────────────────────
# Build Vector Database once at startup
# ─────────────────────────────────────────
def build_vectorstore():
    print("📚 Building RAG knowledge base...")
    all_documents = []

    # load text files
    text_files = [
        "documents/sample.txt",
        "documents/notes.txt"
    ]
    for file in text_files:
        try:
            loader = TextLoader(file, encoding="utf-8")
            docs = loader.load()
            all_documents.extend(docs)
            print(f"  ✅ Loaded: {file}")
        except Exception as e:
            print(f"  ❌ Failed: {file} → {e}")

    # load PDFs
    pdf_files = ["documents/sample.pdf"]
    for file in pdf_files:
        try:
            loader = PyPDFLoader(file)
            docs = loader.load()
            all_documents.extend(docs)
            print(f"  ✅ Loaded: {file}")
        except Exception as e:
            print(f"  ❌ Failed: {file} → {e}")

    # split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(all_documents)

    # create embeddings and store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db_crewai"
    )
    print(f"✅ Knowledge base ready with {len(chunks)} chunks!")
    return vectorstore

# build once when module loads
vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# ─────────────────────────────────────────
# RAG Tool Input Schema
# ─────────────────────────────────────────
class RAGToolInput(BaseModel):
    query: str = Field(
        description="The question or query to search in the documents"
    )


# ─────────────────────────────────────────
# RAG Tool
# ─────────────────────────────────────────
class RAGTool(BaseTool):
    name: str = "Document Search Tool"
    description: str = """Search through company documents, PDFs and 
    notes to find relevant information. Use this tool whenever you 
    need to find specific information from the knowledge base.
    Input should be a clear question or search query."""
    args_schema: type[BaseModel] = RAGToolInput

    def _run(self, query: str) -> str:
        try:
            # retrieve relevant chunks
            docs = retriever.invoke(query)

            if not docs:
                return "No relevant information found in the documents."

            # format results
            results = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'unknown')
                page = doc.metadata.get('page', '')
                page_info = f" (page {int(page)+1})" if page != '' else ""
                results.append(
                    f"Source {i+1}: {source}{page_info}\n"
                    f"Content: {doc.page_content}\n"
                )

            return "\n---\n".join(results)

        except Exception as e:
            return f"Error searching documents: {str(e)}"