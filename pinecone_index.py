import pinecone # To store embeddings
import os # To access env variables
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer # to use NLP models

load_dotenv()

# Fetching Pinecone API key and index name from env variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

model = SentenceTransformer('sentence-t5-large') # Loading model 'all-mpnet-base-v2' model for sentence embedding

def initialize_pinecone():
    """Initializes Pinecone."""
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names(): # New index will get created if the index is not exist
        # creating new Pinecone index with cosine similarity and 768 dimensions.
        pc.create_index(name=PINECONE_INDEX_NAME, metric="cosine", dimension=1024)
    return pc.Index(PINECONE_INDEX_NAME) # Returning initialized pinecone index

def store_query(query: str, index: pinecone.Index):
    """Stores the user query in Pinecone with metadata."""
    # Query ko sentence embedding model se encode kar rahe hain taake uska vector representation mile.
    embedding = model.encode([query])[0].tolist()
    # Index me unique ID generate karne ke liye pehle se stored vectors ki count le rahe hain.
    stats = index.describe_index_stats()
    vector_count = stats['total_vector_count']
    index.upsert(vectors=[(str(vector_count), embedding, {"original_text": query})])
    # Upsert function ka use karke query ko index me store kar rahe hain. Metadata me original text bhi save ho raha hai.

def retrieve_queries(index: pinecone.Index, top_k: int = 5):
    """Retrieves recent user queries from Pinecone with metadata."""
    try:
        stats = index.describe_index_stats()
        total_vectors = stats['total_vector_count'] # Total stored vectors ki count nikal rahe hain.
        start_index = max(0, total_vectors - top_k) # Last ke 'top_k' vectors retrieve karne ke liye start index nikal rahe hain.
        results = index.fetch(ids=[str(i) for i in range(start_index, total_vectors)]) # Last ke 'top_k' queries fetch kar rahe hain.
        # Agar vectors exist karte hain toh unka original text return kar rahe hain.
        if results.vectors: 
            return [vector.metadata['original_text'] for vector in results.vectors.values()]
        else:
            return [] # Agar koi data na mile toh empty list return kar rahe hain.
    except Exception as e:
        print(f"Error retrieving queries: {e}")
        return [] # Error case me empty list return kar rahe hain.