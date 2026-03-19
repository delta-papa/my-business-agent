from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.groq import Groq
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
import os

# Configure models
Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=os.environ.get("GROQ_API_KEY"))
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Load the persisted index
storage_context = StorageContext.from_defaults(persist_dir="storage")
index = load_index_from_storage(storage_context)

SYSTEM_PROMPT = """You are a helpful assistant for Riverside Planetarium.
Only answer questions using the information provided to you about the planetarium.
If a question is not related to the planetarium, politely say:
'I can only help with questions about Riverside Planetarium.'
Do not make up information. If you don't know the answer, say so."""

query_engine = index.as_query_engine(
    similarity_top_k=3,
    system_prompt=SYSTEM_PROMPT
)

print("Riverside Planetarium Bot ready. Type 'quit' to exit.\n")

while True:
    question = input("You: ").strip()
    if question.lower() in ("quit", "exit"):
        break
    if not question:
        continue
    response = query_engine.query(question)
    print(f"\nBot: {response}\n")