import os
import gradio as gr
import fitz  # PyMuPDF for PDF processing
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers

def extract_text_from_pdf(pdf_path, max_pages=5):
    """Extracts text from a given PDF file up to a specified number of pages."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc[:max_pages]])
    return text

def summarize_document(pdf_file):
    """Summarizes the content of a PDF document using a local LLM with RAG."""
    pdf_path = pdf_file.name
    document_text = extract_text_from_pdf(pdf_path)
    
    # Text Splitting
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_text(document_text)
    
    # Embeddings & Vector Storage
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_texts(chunks, embedding=embedding_model, persist_directory="./vector_db")
    
    # LLM & Retrieval-Augmented Generation
    llm = CTransformers(model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_type="mistral", device="cuda")
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type="stuff")
    
    # Generate Summary
    summary = qa_chain.run("Provide a simple summary of this document.")
    return summary

# Gradio UI Setup
iface = gr.Interface(fn=summarize_document,
                     inputs=gr.File(label="Upload PDF"),
                     outputs=gr.Textbox(label="Generated Summary"))

# Launch the Gradio App
iface.launch(share=True)
