import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader, TextLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import re

# --- Simple Math Cleaner (formatting only; no model logic changes) ---

def format_math_for_streamlit(text: str) -> str:
    """Normalize math delimiters so Streamlit renders equations more reliably."""
    if not text:
        return text

    cleaned_text = text
    # Replace block math brackets with $$
    cleaned_text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", cleaned_text, flags=re.DOTALL)
    # Replace inline math brackets with $
    cleaned_text = re.sub(r"\\\((.*?)\\\)", r"$\1$", cleaned_text, flags=re.DOTALL)
    # Catch plain brackets used for block equations
    cleaned_text = re.sub(r"^\s*\[\s*(.*?)\s*\]\s*$", r"$$\1$$", cleaned_text, flags=re.MULTILINE | re.DOTALL)

    return cleaned_text

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

class Chunking:
    def __init__(self, file_path):
        self.file_path = file_path
        self.docs = []
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

    def load_file(self):
        extension = os.path.splitext(self.file_path)[1].lower()
        if extension == ".txt":
            self.docs = TextLoader(self.file_path).load()
        elif extension == ".csv":
            self.docs = CSVLoader(self.file_path).load()
        elif extension == ".pdf":
            self.docs = PyMuPDFLoader(self.file_path).load()
        else:
            st.error("Unsupported file format")
            return None
        return self.docs

    def create_chunks(self):
        if not self.docs:
            return []
        return self.text_splitter.split_documents(self.docs)

class VectorDatabase:
    def __init__(self, documents):
        self.documents = documents

    def create_embeddings(self):
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        return Chroma.from_documents(self.documents, embeddings)

# --- 1. Session State Initialization ---
# This ensures variables survive when Streamlit reruns the script
if "messages" not in st.session_state:
    st.session_state.messages = [] # Stores chat history

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None # Stores the embedded documents

# --- 2. Sidebar for File Uploading ---
# We use a button to trigger the embedding process so it only happens ONCE
with st.sidebar:
    st.header("Document Knowledge")
    uploaded_file = st.file_uploader("Upload your file here", type=['pdf', 'txt', 'csv'])
    
    if st.button("Process Document"):
        if uploaded_file is not None:
            with st.spinner("Chunking and Embedding Document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                chunking_obj = Chunking(tmp_path)
                if chunking_obj.load_file():
                    documents = chunking_obj.create_chunks()
                    embedding_obj = VectorDatabase(documents)
                    
                    # Save the database into session_state!
                    st.session_state.vector_db = embedding_obj.create_embeddings()
                    st.success("Document processed! You can now ask questions about it.")
                
                os.remove(tmp_path)
        else:
            st.warning("Please upload a file first.")
            
    if st.button("Clear Chat History"):
        st.session_state.messages = []

# --- 3. Main UI and Chat Logic ---
st.title("Conversational AI & RAG")

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Wait for user input
if prompt := st.chat_input("Ask a question (with or without a document)..."):
    
    # Immediately display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            # Initialize the LLM (Using the updated, supported Groq model)
            llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_api_key)
            
            # Reconstruct conversational memory for LangChain
            chat_history_objects = []
            for msg in st.session_state.messages[:-1]: # All messages except the current prompt
                if msg["role"] == "user":
                    chat_history_objects.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history_objects.append(AIMessage(content=msg["content"]))


            system_prompt_text = """You are a highly intelligent, versatile, and friendly AI assistant. Your goal is to provide accurate, well-structured, and engaging answers.

In this mode:
- Use your internal reasoning and knowledge.
- Be transparent that the response is based on general knowledge.

### MATH FORMATTING RULES (CRITICAL):
1. TRANSLATION: You MUST actively translate ugly math text into proper LaTeX.
2. DELIMITERS: You are STRICTLY FORBIDDEN from using `\[ \]`, `\( \)`, or plain `[ ]` to wrap equations. 
3. INLINE MATH: Wrap all variables and inline math exclusively in single dollar signs (e.g., $d_{{model}} = 512$).
4. BLOCK MATH: Wrap all standalone equations exclusively in double dollar signs.
Example:
$$
\text{{FFN}}(x) = \text{{ReLU}}(x W_1 + b_1) W_2 + b_2
$$
"""
            
            # ==========================================
            # 2. RAG OVERRIDE PROMPT (DOCUMENT UPLOADED)
            # ==========================================
            if st.session_state.vector_db is not None:
                retriever = st.session_state.vector_db.as_retriever()
                relevant_docs = retriever.invoke(prompt)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # IMPORTANT: Notice the 'f' before the triple quotes! 
                # IMPORTANT: Notice the double braces around {{model}} so Python doesn't crash!
                system_prompt_text = f"""You are an advanced Retrieval-Augmented Generation (RAG) AI assistant.

========================
CORE BEHAVIOR
========================
1. First, attempt to answer the user's question using ONLY the provided context.
2. If the provided context contains the answer, prioritize it over prior knowledge and do not hallucinate.
3. HYBRID FALLBACK: If the answer is NOT available in the provided context, you must explicitly say: "I could not find this specific information in the provided document, but based on my general knowledge..." and then answer the question to the best of your ability.

### MATH FORMATTING RULES (CRITICAL):
1. TRANSLATION: The provided context may contain poorly formatted math extracted from a PDF (e.g., 'h×dk=8×64'). You MUST actively translate these into proper, beautifully formatted LaTeX.
2. DELIMITERS: You are STRICTLY FORBIDDEN from using `\[ \]`, `\( \)`, or plain `[ ]` to wrap equations. 
3. INLINE MATH: Wrap all variables and inline math exclusively in single dollar signs (e.g., $d_{{model}} = 512$).
4. BLOCK MATH: Wrap all standalone equations exclusively in double dollar signs.
Example:
$$
\text{{FFN}}(x) = \text{{ReLU}}(x W_1 + b_1) W_2 + b_2
$$

Context:
{context}
"""
            
            # Build final prompt package: System constraints + Memory + Current Question
            final_messages = [
                SystemMessage(content=system_prompt_text),
                *chat_history_objects,
                HumanMessage(content=prompt)
            ]
            
            # Generate and display response
            response = llm.invoke(final_messages)

            # --- Simple Math Cleaner ---
            cleaned_text = format_math_for_streamlit(response.content)

            st.markdown(cleaned_text)

            # Save the AI's cleaned response to session state memory
            st.session_state.messages.append({"role": "assistant", "content": cleaned_text})