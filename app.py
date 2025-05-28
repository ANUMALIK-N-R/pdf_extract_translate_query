import streamlit as st
from utils.pdf_utils import extract_text_from_pdf
from utils.language_utils import detect_language, translate_text
from utils.qa_utils import init_vector_store, answer_question

st.set_page_config(page_title="Multilingual PDF Assistant", layout="wide")

st.title("📄 Multilingual PDF Assistant")

with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    target_lang = st.text_input("Translate to (e.g., en, fr, de):", "en")

    qa_source = st.radio(
        "Use text for Q&A from:",
        options=["Original text", "Translated text"],
        index=0,
        help="Choose if the Q&A system should use the original extracted text or the translated text"
    )

# Session state for texts and Q&A status
if "original_text" not in st.session_state:
    st.session_state["original_text"] = None
if "translated_text" not in st.session_state:
    st.session_state["translated_text"] = None
if "qa_initialized" not in st.session_state:
    st.session_state["qa_initialized"] = False

if uploaded_file is None:
    st.info("Please upload a PDF file from the sidebar to get started.")
else:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    if st.session_state["original_text"] is None:
        with st.spinner("Extracting text from PDF..."):
            st.session_state["original_text"] = extract_text_from_pdf("temp.pdf")
        st.success("Text extracted!")

    text = st.session_state["original_text"]

    if text:
        with st.expander("📋 Extracted Text Preview"):
            st.text_area("Extracted Text", text[:3000], height=300)

        detected_lang = detect_language(text)
        st.markdown(f"🔍 **Detected Language:** {detected_lang}")

        # Translate button
        if st.button("Translate"):
            with st.spinner(f"Translating to {target_lang}..."):
                translated = translate_text(text, src_lang=detected_lang, tgt_lang=target_lang)
                st.session_state["translated_text"] = translated
            st.success("Translation complete!")

        # Show translated text if available
        if st.session_state["translated_text"]:
            with st.expander(f"🌐 Translated Text Preview ({target_lang})"):
                st.text_area("Translated Text", st.session_state["translated_text"][:3000], height=300)

        # Initialize Q&A system button uses selected text source
        if st.button("Initialize Q&A System"):
            source_text = (
                st.session_state["translated_text"]
                if qa_source == "Translated text" and st.session_state["translated_text"] is not None
                else st.session_state["original_text"]
            )
            with st.spinner("Initializing Q&A system..."):
                init_vector_store(source_text)
                st.session_state["qa_initialized"] = True
            st.success(f"Q&A system initialized on {qa_source.lower()}!")

        # Question input and answering only if Q&A initialized
        if st.session_state["qa_initialized"]:
            query = st.text_input("Ask a question about the document:")
            query = st.text_input("Ask a question about the document:")
        if query:
            answer = answer_question(query, target_lang=target_lang, source_lang=lang)
            st.write(f"🤖 Answer: {answer}")


    else:
        st.error("Failed to extract text from the PDF.")
