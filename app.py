import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
# This is the updated import line that fixes the error:
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import os

# Page Config
st.set_page_config(page_title="DocuChat AI", layout="wide")

# Title
st.title("🤖 Chat with your Documents")

# Sidebar for API Key and File Upload
with st.sidebar:
    st.header("Settings")
    
    # Check for secrets first, otherwise use input
    if 'GOOGLE_API_KEY' in st.secrets:
        api_key = st.secrets['GOOGLE_API_KEY']
    else:
        api_key = st.text_input("Enter Google API Key", type="password")

    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Main App Logic
if api_key and uploaded_file:
    # Set the API Key
    os.environ["GOOGLE_API_KEY"] = api_key

    # 1. Process the PDF (Cached)
    @st.cache_resource(show_spinner=False)
    def process_pdf(file):
        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file.read())
                temp_path = temp_file.name

            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            # Using the fixed text splitter import
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = FAISS.from_documents(splits, embeddings)
            return vectorstore

    vectorstore = process_pdf(uploaded_file)
    retriever = vectorstore.as_retriever()

    # 2. Setup the LLM Chain
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 3. Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"input": user_question})
                answer = response['answer']
                st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

elif not api_key:
    st.warning("Please enter your Google API Key to proceed.")
elif not uploaded_file:
    st.info("Please upload a PDF document to start chatting.")
