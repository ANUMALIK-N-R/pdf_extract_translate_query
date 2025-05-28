import streamlit as st
from utils.pdf_utils import extract_text_from_pdf
from utils.language_utils import detect_language, translate_text
from utils.qa_utils import init_vector_store, answer_question

st.set_page_config(page_title="Multilingual PDF Assistant", layout="wide")

st.title("📄 Multilingual PDF Assistant")

# Sidebar for upload and language selection
with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    target_lang = st.text_input("Translate to (e.g., en, fr, de):", "en")

# Main page
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

        # Button to initialize Q&A and translate text
        if st.button("Initialize Q&A System"):
            with st.spinner("Initializing Q&A system and translating text..."):
                init_vector_store(text)
                translated = translate_text(text, src_lang=lang, tgt_lang=target_lang)
            st.success("Q&A system initialized!")

            with st.expander(f"🌐 Translated Text Preview ({target_lang})"):
                st.text_area("Translated Text", translated[:3000], height=300)

            # Show question input and answer area only after initialization
            query = st.text_input("Ask a question about the document:")
            if query:
                with st.spinner("Searching for answer..."):
                    answer = answer_question(query, target_lang=target_lang)
                st.markdown(f"🤖 **Answer:** {answer}")
        else:
            st.info("Initialize the Q&A system to translate and ask questions.")
    else:
        st.error("Failed to extract text from the PDF.")
