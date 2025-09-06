#  I have used  FAISS as a vectordb becuase it is a library for efficient similarity search and clustering of dense vectors
#importing necessary libraries
import faiss # type: ignore 
import pickle
import tempfile
from PyPDF2 import PdfReader  #type: ignore
from openai_helpers import embed_texts
import numpy as np

CHUNK_SIZE = 500

def read_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def build_or_load_vectorstore(client, uploaded_file):
    text = read_pdf(uploaded_file)
    chunks = chunk_text(text)

    embeddings = embed_texts(client, chunks)

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    metadata = {"chunks": chunks}

    # Optionally save index
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        pickle.dump((index, metadata), f)

    return index, metadata

def query_vectorstore(index, query, client, top_k=3):
    query_embedding = embed_texts(client, [query])[0]
    D, I = index.search(np.array([query_embedding]).astype("float32"), top_k)
    return " ".join([index_to_context(i, metadata) for i in I[0]])  

def index_to_context(i, metadata):
    return metadata["chunks"][i] if i < len(metadata["chunks"]) else ""

