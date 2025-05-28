import streamlit as st
from utils.pdf_utils import extract_text_from_pdf
from utils.language_utils import detect_language, translate_text
from utils.qa_utils import init_vector_store, answer_question

st.set_page_config(page_title="Multilingual PDF Assistant", layout="wide")

st.title("📄 Multilingual PDF Assistant")

# Sidebar for upload and language input
with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    target_lang = st.text_input("Translate to (e.g., en, fr, de):", "en")

# Initialize session state for translated text and Q&A
if "translated_text" not in st.session_state:
    st.session_state["translated_text"] = None
if "qa_initialized" not in st.session_state:
    st.session_state["qa_initialized"] = False

if uploaded_file is None:
    st.info("Please upload a PDF file from the sidebar to get started.")
else:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf("temp.pdf")

    if text:
        with st.expander("📋 Extracted Text Preview"):
            st.text_area("Extracted Text", text[:3000], height=300)

        lang = detect_language(text)
        st.markdown(f"🔍 **Detected Language:** {lang}")

        # Translate button (works anytime)
        if st.button("Translate"):
            with st.spinner(f"Translating to {target_lang}..."):
                translated = translate_text(text, src_lang=lang, tgt_lang=target_lang)
                st.session_state["translated_text"] = translated
            st.success("Translation complete!")

        # Show translated text preview if available
        if st.session_state["translated_text"]:
            with st.expander(f"🌐 Translated Text Preview ({target_lang})"):
                st.text_area("Translated Text", st.session_state["translated_text"][:3000], height=300)

        # Initialize Q&A system button
        if st.button("Initialize Q&A System"):
            with st.spinner("Initializing Q&A system..."):
                init_vector_store(text)
                st.session_state["qa_initialized"] = True
            st.success("Q&A system initialized! You can now ask questions below.")

        # Question input and answer (only if Q&A initialized)
        if st.session_state["qa_initialized"]:
            query = st.text_input("Ask a question about the document:")
            if query:
                with st.spinner("Searching for answer..."):
                    answer = answer_question(query, target_lang=target_lang)
                st.markdown(f"🤖 **Answer:** {answer}")

    else:
        st.error("Failed to extract text from the PDF.")
