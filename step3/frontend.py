import streamlit as st 
import requests  # Correcto con 's' al final

BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Chat with your PDFs",
    page_icon="📚",
    layout="wide"
)

st.title("Chat with your PDFs")
st.caption("Powered by RAG - Retrieval Augmented Generation")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- BARRA LATERAL (Sidebar) ---
with st.sidebar:
    st.header("Your Documents")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload any PDF and start asking questions"
    )

    if uploaded_file:
        if st.button("Ingest PDF", type="primary", use_container_width=True):
            with st.spinner("Reading, chunking and embedding your PDF..."):
                response = requests.post(
                    f"{BACKEND_URL}/ingest", 
                    files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                )
            if response.status_code == 200:
                data = response.json()
                st.success(f"Done! Added {data['chunks_added']} chunks")
                st.info(f"Total in database: {data['total_chunks']} chunks")
            else:
                st.error("Something went wrong. Is the backend running?")
                
    st.divider()

    # Control de cuántos fragmentos recuperar (Soluciona el NameError de n_results)
    n_results = st.slider("Fragmentos a recuperar (n_results)", min_value=1, max_value=10, value=3)

    st.divider()

    try:
        status = requests.get(f"{BACKEND_URL}/").json()
        st.metric("chunks in database", status["total_chunks"])
    except:
        st.error("Backend is not reachable")

# --- PANTALLA PRINCIPAL  ---

for msg in st.session_state.messages:  
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        
        if "sources" in msg and msg["sources"]:
            with st.expander("View retrieved chunks", expanded=False):
                for source in msg["sources"]:
                    st.markdown(
                        f"**{source['source']}** - Page {source['page']} "
                        f"*(similarity score: {source['score']})*"
                    )

if question := st.chat_input("Ask anything about your documents"):
    # Guardar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": question, "n_results": n_results}
                )    
                data = response.json()

                if "error" in data:
                    answer = f"{data['error']}"
                    sources = []
                else:
                    answer = data["answer"]
                    sources = data["sources"]
            except Exception as e:
                answer = f"Could not reach backend: {e}"
                sources = []

            st.write(answer)
            
            if sources:
                with st.expander("View retrieved chunks", expanded=False):
                    for source in sources:
                        st.markdown(
                            f"**{source['source']}** - Page {source['page']} "
                            f"*(similarity score: {source['score']})*"
                        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
