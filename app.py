
#importing necessary libraries
import streamlit as st #for wed app 
from openai_helpers import get_openai_client, ask_llm  # for opeanai apis
from vectorstore import build_or_load_vectorstore, query_vectorstore 

# Setting up the title and layout of the app
st.set_page_config(page_title="PDF Q&A App", layout="wide")
st.title("📄 PDF Q&A Chatbot")

#Code for file upload and processing
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    st.success("✅ File uploaded successfully!")

    # Process PDF only once
    if "vector_index" not in st.session_state:  # to store index in a sessio
        with st.spinner("Processing PDF and creating embeddings..."):
            client = get_openai_client()
            index, metadata = build_or_load_vectorstore(client, uploaded_file)
            st.session_state["vector_index"] = index
            st.session_state["metadata"] = metadata
        st.success("✅ PDF processed successfully!")

    # Chat interface
    st.subheader("Ask questions about your PDF:")
    user_question = st.text_input("Enter your question")

    if st.button("Ask") and user_question:
        client = get_openai_client()
        index = st.session_state["vector_index"]

        # Retrieve relevant context
        context = query_vectorstore(index, user_question, client)

        # Code for generating answers usin LLM
        with st.spinner("Generating answer..."):
            answer = ask_llm(client, user_question, context)

        st.markdown("### 🤖 Answer:")
        st.write(answer)

