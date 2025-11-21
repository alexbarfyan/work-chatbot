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
st.set_page_config(page_title="Resume Bot", layout="wide")
st.title("🤖 Chat with my Resume")

# --- SMART FILE SEARCH ---
def find_pdf():
    current_files = os.listdir('.')
    for f in current_files:
        if f.lower().endswith(".pdf"):
            return os.path.join('.', f)
    if os.path.exists('..'):
        parent_files = os.listdir('..')
        for f in parent_files:
            if f.lower().endswith(".pdf"):
                return os.path.join('..', f)
    return None

pdf_path = find_pdf()

# --- API KEY CHECK ---
if 'GOOGLE_API_KEY' in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets['GOOGLE_API_KEY']
else:
    st.error("❌ Google API Key is missing.")
    st.stop()

# --- PROCESSING ENGINE (With Progress Bar) ---
@st.cache_resource(show_spinner=False)
def setup_knowledge_base(path):
    # This "Status" widget looks professional and explains the wait time
    with st.status("🚀 Booting up the AI Assistant...", expanded=True) as status:
        
        st.write("📥 Downloading Neural Network Model (all-MiniLM-L6-v2)...")
        # The heavy download happens here
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        st.write("📄 Reading Document...")
        loader = PyPDFLoader(path)
        docs = loader.load()
        
        st.write("✂️ Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        st.write("🧠 Building Vector Index...")
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        status.update(label="✅ System Ready!", state="complete", expanded=False)
        return vectorstore

if not pdf_path:
    st.error("❌ No PDF found in the repository.")
    st.stop()

# Run the setup
vectorstore = setup_knowledge_base(pdf_path)

if vectorstore:
    retriever = vectorstore.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    
    system_prompt = (
        "You are a helpful assistant answering questions about the resume provided. "
        "Answer based strictly on the context. "
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("Ask about my experience..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = rag_chain.invoke({"input": user_question})
                    answer = response['answer']
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
