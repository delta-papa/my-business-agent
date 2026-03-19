import streamlit as st
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.groq import Groq
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=os.environ.get("GROQ_API_KEY"))
#Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

storage_context = StorageContext.from_defaults(persist_dir="storage")
index = load_index_from_storage(storage_context)

SYSTEM_PROMPT = """You are a helpful assistant for Riverside Planetarium.
Only answer questions using the information provided to you about the planetarium.
If a question is not related to the planetarium, politely say:
'I can only help with questions about Riverside Planetarium.'
Do not make up information. If you don't know the answer, say so."""

query_engine = index.as_query_engine(similarity_top_k=3, system_prompt=SYSTEM_PROMPT)

st.title("Riverside Planetarium Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about our shows, hours, tickets..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        response = query_engine.query(prompt)
        st.write(str(response))
    st.session_state.messages.append({"role": "assistant", "content": str(response)})