# app.py
from load_docs import load_documents,chunk_documents
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# 1. Title of the app
st.title("📄 Tom’s RAGtime Show 🎤📚🤖")

# 2. File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Load PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    documents = load_documents("temp.pdf")
    # 2. Split into chunks
    chunks = chunk_documents(documents)

    # Create embeddings + vector DB
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vectorstore = Chroma.from_documents(chunks, embeddings)

    # Build HuggingFace LLM
    hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", device=0)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # RetrievalQA chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # 3. Input box for user query
    query = st.text_input("Ask a question about the document:")

    # 4. Generate answer
    if query:
        with st.spinner("Generating answer..."):
            answer = qa_chain.run(query)
        st.write("**Answer:**", answer)