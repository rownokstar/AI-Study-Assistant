# 🤖 AI Study Assistant

An intelligent chatbot designed to help users study more effectively by answering questions from and generating summaries of PDF documents. This project is built with LangChain, OpenAI, and Streamlit.

## ✨ Key Features

-   **Interactive Q&A:** Ask questions in natural language about the content of your uploaded PDFs.
-   **Full Document Summarization:** Generate a concise summary of the entire document with a single click using the map-reduce technique.
-   **Multi-Document Support:** Upload and process multiple PDF files at once.
-   **User-Friendly Interface:** Built with Streamlit for a clean and intuitive user experience.

## 🚀 Live Demo

https://colab.research.google.com/drive/1haC0qZAdTGp03gzclUqAMp6_c0yeIzvI?usp=sharing

## 🛠️ Technologies Used

-   **Backend & AI Logic:** Python, LangChain, OpenAI API
-   **Frontend:** Streamlit
-   **Vector Storage:** FAISS
-   **Deployment:** Google Colab, Ngrok

## ⚙️ How to Use

1.  Open the Google Colab notebook.
2.  Add your OpenAI and Ngrok keys to the Colab Secrets Manager.
3.  Run all the cells in order.
4.  Click the public `ngrok` URL to open the app.
5.  Upload your PDF(s), click 'Process Documents', and start asking questions!
