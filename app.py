import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

# Page Config
st.set_page_config(page_title="Team Knowledge Bot", layout="wide")

st.title("🤖 Team Knowledge Assistant")
st.markdown("I have already read the document. Ask me anything!")

# --- CONFIGURATION ---
# We look for the secret key. If not found, we stop.
if 'GOOGLE_API_KEY' in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets['GOOGLE_API_KEY']
else:
    st.error("❌ Google API Key is missing in Streamlit Secrets.")
    st.stop()

# --- LOADING THE DOC ---
@st.cache_resource(show_spinner=False)
def load_and_process_document():
    file_path = "context.pdf" # This matches the file name you uploaded to GitHub
    
    if not os.path.exists(file_path):
        st.error(f"❌ Could not find {file_path} in the repository.")
        return None

    with st.spinner("Waking up and loading knowledge base..."):
        # Load
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Embed (Local Memory)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore

# Initialize the knowledge base once
vectorstore = load_and_process_document()

if vectorstore:
    retriever = vectorstore.as_retriever()

    # --- BRAIN SETUP ---
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    
    system_prompt = (
        "You are an expert assistant representing the document provided. "
        "Answer questions based ONLY on the following context. "
        "If the answer is not in the context, say 'I am not sure, it isn't in the document.' "
        "Keep answers professional and concise."
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

    # --- CHAT INTERFACE ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input
    if user_question := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Checking the document..."):
                response = rag_chain.invoke({"input": user_question})
                answer = response['answer']
                st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
