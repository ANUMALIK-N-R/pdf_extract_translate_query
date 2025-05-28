import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from utils.language_utils import translate_text, detect_language

nltk.download('punkt_tab')

chunks = []
embeddings = None
model = SentenceTransformer("all-MiniLM-L6-v2")

def split_text(text, max_length=500):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def init_vector_store(text):
    global chunks, embeddings
    chunks = split_text(text)
    embeddings = model.encode(chunks, convert_to_tensor=True)


def answer_question(query, target_lang="en", source_lang="en"):
    global embeddings, chunks
    if embeddings is None:
        return "Please initialize the Q&A system first."
    
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings, top_k=3)
    context = " ".join([chunks[hit["corpus_id"]] for hit in hits[0]])
    
    # Dummy answer generation based on context (you can improve this logic)
    answer = f"Based on context: {context[:300]}..."

    # Translate the answer to target_lang if different from source_lang
    if source_lang != target_lang:
        answer = translate_text(answer, src_lang=source_lang, tgt_lang=target_lang)
    
    return answer
