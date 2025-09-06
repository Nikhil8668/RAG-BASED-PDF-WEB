

# Models used for embeddings and for chat purposees 
#Note : These modells are not free as i dont hace openai api access
#  o i have just mentioned the models as it will not work without api access
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o-mini"

# Chunking and retrieval
CHUNK_SIZE = 1200       # ~ words per chunk
CHUNK_OVERLAP = 150     # overlap words
TOP_K = 5               # retrieve top-k chunks

# FAISS index storage
INDEX_DIR = ".rag_index"
