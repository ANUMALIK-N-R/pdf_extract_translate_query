from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer, AutoConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

def model_exists(model_name):
    """
    Checks if a model exists on Hugging Face Hub by trying to load its config.
    Returns True if exists, False otherwise.
    """
    try:
        AutoConfig.from_pretrained(model_name)
        return True
    except Exception:
        return False

def translate_text(text, src_lang="auto", tgt_lang="en", max_chunk_chars=500):
    if src_lang == "auto":
        src_lang = detect(text)
    src_lang = src_lang.lower()
    tgt_lang = tgt_lang.lower()

    # Determine model name
    model_name = None

    # Priority list of models to try
    candidates = []

    # Add direct src-tgt pair first
    candidates.append(f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}")

    # Add multilingual source → English if tgt_lang is English
    if tgt_lang == "en":
        candidates.append("Helsinki-NLP/opus-mt-mul-en")

    # Add English → multilingual target if src_lang is English
    if src_lang == "en":
        candidates.append("Helsinki-NLP/opus-mt-en-mul")

    # Iterate candidates and pick the first existing model
    for candidate in candidates:
        if model_exists(candidate):
            model_name = candidate
            break

    if model_name is None:
        return f"Translation model for {src_lang}→{tgt_lang} not found."

    # Load model and tokenizer
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Split text into chunks (rough char limit)
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 2 <= max_chunk_chars:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    translated_chunks = []
    for chunk in chunks:
        tokens = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**tokens)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_chunks.append(translated_text)

    return " ".join(translated_chunks)
