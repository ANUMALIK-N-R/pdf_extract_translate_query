from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer

def detect_language(text):
    return detect(text)

from transformers import MarianMTModel, MarianTokenizer

def translate_text(text, src_lang="auto", tgt_lang="en", max_length=512):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Split text into manageable chunks
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk + sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Translate each chunk
    translated_chunks = []
    for chunk in chunks:
        tokens = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**tokens)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_chunks.append(translated_text)

    return " ".join(translated_chunks)
