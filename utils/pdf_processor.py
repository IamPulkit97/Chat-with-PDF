import os
from typing import List, Dict, Tuple
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Hybrid PDF processor:
        - Extracts text with page numbers
        - Splits into semantic chunks
        - Embeds chunks
        - Stores in FAISS vector store
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def extract_text_from_pdf(self, pdf_file) -> List[Tuple[str, int]]:
        """Extract text from PDF with page numbers"""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_with_pages = []

        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text and text.strip():
                text_with_pages.append((text, page_num + 1))

        return text_with_pages

    def chunk_text(self, text_with_pages: List[Tuple[str, int]]) -> List[Dict]:
        """Split text into chunks with metadata"""
        chunks = []

        for text, page_num in text_with_pages:
            text_chunks = self.text_splitter.split_text(text)

            for chunk in text_chunks:
                chunks.append({
                    "text": chunk,
                    "metadata": {
                        "page": page_num,
                        "source": f"Page {page_num}"
                    }
                })

        return chunks

    def build_vector_store(self, chunks: List[Dict]):
        """Embed chunks and store them in FAISS"""
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Convert chunks to LangChain-compatible "documents"
        docs = [
            Document(page_content=chunk["text"], metadata=chunk["metadata"])
            for chunk in chunks
        ]

        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store

    def process_pdf(self, pdf_file):
        """Full pipeline: PDF → Chunks → Embeddings → FAISS"""
        text_with_pages = self.extract_text_from_pdf(pdf_file)
        chunks = self.chunk_text(text_with_pages)
        vector_store = self.build_vector_store(chunks)
        return vector_store
