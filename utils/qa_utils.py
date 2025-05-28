from sentence_transformers import SentenceTransformer, util
from utils.language_utils import translate_text, detect_language

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
    context = " ".join([chunks[hit["corpus_id"]] for hit in hits[0]])
    
    # Translate context to target language
    src_lang = detect_language(context)
    translated_context = translate_text(context, src_lang=src_lang, tgt_lang=target_lang)
    
    return f"Based on context: {translated_context[:300]}..."
