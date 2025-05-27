from sentence_transformers import SentenceTransformer, util

chunks = []
embeddings = None
model = SentenceTransformer("all-MiniLM-L6-v2")

def split_text(text, max_length=500):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def init_vector_store(text):
    global chunks, embeddings
    chunks = split_text(text)
    embeddings = model.encode(chunks, convert_to_tensor=True)

def answer_question(query):
    global embeddings, chunks
    if embeddings is None:
        return "Please initialize the Q&A system first."
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings, top_k=3)
    context = " ".join([chunks[hit["corpus_id"]] for hit in hits[0]])
    return f"Based on context: {context[:300]}..."
