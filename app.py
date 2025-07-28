# Final app.py code for Streamlit Cloud Deployment

import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain

# --- Core Functions ---

def get_pdf_documents(pdf_files):
    documents = []
    temp_dir = "temp_pdf"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    for pdf_file in pdf_files:
        file_path = os.path.join(temp_dir, pdf_file.name)
        with open(file_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    document_chunks = text_splitter.split_documents(documents)
    return document_chunks

def get_vectorstore(document_chunks, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(documents=document_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(temperature=0.3, openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

# --- Streamlit UI ---

def main():
    # --- Page Configuration ---
    st.set_page_config(page_title="AI Study Assistant", page_icon="🤖", layout="wide")

    # --- Session State Initialization ---
    if "conversation" not in st.session_state: st.session_state.conversation = None
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "processed_documents" not in st.session_state: st.session_state.processed_documents = None
    if "summary" not in st.session_state: st.session_state.summary = None

    # Check for API Key in Streamlit Secrets
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("🚨 OpenAI API Key not found! Please add it to your Streamlit Secrets.")
        st.stop()
    
    openai_api_key = st.secrets["OPENAI_API_KEY"]

    # --- Header and Introduction ---
    st.title("AI Study Assistant 🤖")
    st.write("Upload your textbooks or documents, and I'll help you study by answering questions and generating summaries.")
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1.  **Upload:** Use the sidebar to upload one or more PDF documents.
        2.  **Process:** Click 'Process Documents' to let the AI read them.
        3.  **Ask:** Once processing is complete, ask any question about the content in the chat box.
        4.  **Summarize:** Generate a full summary using the button in the sidebar.
        """)
    st.markdown("---")

    # --- Chat Interface ---
    st.header("Chat with Your Documents")
    user_question = st.text_input("Ask a question about the content of your documents:")
    if user_question and st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(f"<div style='text-align: right;'><b>You:</b> {message.content}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<b>Bot:</b> {message.content}")

    # --- Sidebar for Controls ---
    with st.sidebar:
        st.header("Controls")
        st.subheader("1. Upload Documents")
        pdf_docs = st.file_uploader("Upload your PDF files and click 'Process'", accept_multiple_files=True, type="pdf")
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    doc_chunks = get_pdf_documents(pdf_docs)
                    st.session_state.processed_documents = doc_chunks
                    vectorstore = get_vectorstore(doc_chunks, openai_api_key)
                    st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
                    st.success("Processing complete!")
            else:
                st.error("Please upload at least one PDF file.")
        st.markdown("---")
        st.subheader("2. Additional Features")
        if st.button("Generate Full Summary"):
            if st.session_state.processed_documents:
                with st.spinner("Generating summary..."):
                    llm = ChatOpenAI(temperature=0.2, openai_api_key=openai_api_key)
                    summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
                    summary = summary_chain.run(st.session_state.processed_documents)
                    st.session_state.summary = summary
            else:
                st.warning("Please process your documents first.")
        if st.session_state.summary:
            with st.expander("View Document Summary"):
                st.write(st.session_state.summary)
 # --- Footer ---
    st.markdown("---")
    st.markdown(
        '<h6>Programmed & Developed by <a href="https://www.linkedin.com/in/dm-shahriar-hossain/" target="_blank">D.M. Shahriar Hossain</a></h6>',
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
