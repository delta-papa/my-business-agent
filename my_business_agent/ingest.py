from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

# Point to your local Ollama model
Settings.llm = Ollama(model="llama3.2", request_timeout=60.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
#Settings.embed_model = OllamaEmbedding(model_name="llama3.2")


# Load your knowledge base
documents = SimpleDirectoryReader("data").load_data()
print(f"Loaded {len(documents)} document(s)")

# Build the index and persist it to disk
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir="storage")
print("Index built and saved to ./storage")