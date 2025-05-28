import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from utils.language_utils import translate_text, detect_language

nltk.download('punkt')

chunks = []
embeddings = None
model = SentenceTransformer("all-MiniLM-L6-v2")

def split_text(text, max_length=500):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def init_vector_store(text):
    global chunks, embeddings
    chunks = split_text(text)
    embeddings = model.encode(chunks, convert_to_tensor=True)

def answer_question(query, target_lang="en"):
    global embeddings, chunks
    if embeddings is None:
        return "Please initialize the Q&A system first."

    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings, top_k=3)

    # Take the top relevant chunk only
    top_chunk_id = hits[0][0]["corpus_id"]
    top_chunk = chunks[top_chunk_id]

    # Split chunk into sentences
    sentences = sent_tokenize(top_chunk)

    # Compute embeddings for sentences
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    # Calculate cosine similarities with query
    scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
    best_idx = scores.argmax()
    best_sentence = sentences[best_idx]

    # Detect source language and translate to target language
    src_lang = detect_language(best_sentence)
    translated_answer = translate_text(best_sentence, src_lang=src_lang, tgt_lang=target_lang)

    return translated_answer
