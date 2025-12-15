
import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA



st.set_page_config(page_title="Chat with Documents")

st.title("📄 Chat with Documents")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{uploaded_file.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    st.success("Document uploaded and processed successfully!")

    query = st.text_input("Ask a question from the document")

    if query:
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(),
            retriever=vectorstore.as_retriever()
        )
        answer = qa.run(query)

        st.subheader("Answer")
        st.write(answer)
