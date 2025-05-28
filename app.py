import streamlit as st
from utils.pdf_utils import extract_text_from_pdf
from utils.language_utils import detect_language, translate_text
from utils.qa_utils import init_vector_store, answer_question

st.set_page_config(page_title="Multilingual PDF Assistant", layout="wide")

st.title("📄 Multilingual PDF Assistant")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf("temp.pdf")
        st.success("Text extracted successfully!")
        st.text_area("Extracted Text", text[:3000], height=300)

    if text:
        lang = detect_language(text)
        st.write(f"🔍 Detected Language: **{lang}**")

        target_lang = st.text_input("Translate to (e.g., en, fr, de):", "en")
        if st.button("Translate"):
            translated = translate_text(text, src_lang=lang, tgt_lang=target_lang)
            st.text_area("Translated Text", translated[:3000], height=300)

        if st.button("Initialize Q&A System"):
            init_vector_store(text)
            st.success("Q&A System initialized. Ask questions below!")

        query = st.text_input("Ask a question about the document:")
        if query:
            answer = answer_question(query, target_lang=target_lang)
            st.write(f"🤖 Answer: {answer}")

