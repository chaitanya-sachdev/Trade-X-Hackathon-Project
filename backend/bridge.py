import os
import sys

# Direct imports - since 'success' was printed in terminal, these will work.
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- CONFIGURATION ---
DB_DIR = "./trade_x_vdb"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class TradeDataArchitect:
    def __init__(self):
        print(f"[*] Initializing CPU-bound Embeddings ({EMBED_MODEL_NAME})...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\nChapter", "\n\nSection", "\nNote:", "\n\n", "\n", " "]
        )

    def ingest_document(self, file_path, country_code, year):
        if not os.path.exists(file_path):
            print(f"[!] Error: File '{file_path}' not found.")
            return None
            
        print(f"[*] Starting Ingestion: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        chunks = self.splitter.split_documents(docs)
        
        for chunk in chunks:
            chunk.metadata.update({
                "country_code": country_code,
                "year": int(year),
                "source": os.path.basename(file_path),
                "hs_chapter": self._extract_chapter(chunk.page_content)
            })
            
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=DB_DIR
        )
        print(f"[+] Ingestion Complete: {len(chunks)} chunks indexed to {DB_DIR}")
        return vector_db

    def _extract_chapter(self, text):
        text_start = text[:150].lower()
        if "chapter" in text_start:
            try:
                words = text_start.split("chapter")[1].strip().split()
                chapter_num = ''.join(filter(str.isdigit, words[0]))
                return chapter_num if chapter_num else "General"
            except:
                pass
        return "General"

    def query_trade_law(self, query, country_code, chapter=None):
        db = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)
        search_filter = {"country_code": country_code}
        if chapter:
            search_filter["hs_chapter"] = str(chapter)
            
        print(f"[*] Querying '{query}' for Country: {country_code}...")
        results = db.similarity_search(query, k=3, filter=search_filter)
        return results
    def setup_correction_cache(self):
        """Initializes the Tier 1 Cache collection."""
        # We use Cosine Distance for the cache to find 'near-exact' string matches
        self.cache_collection = Chroma(
            collection_name="correction_cache",
            embedding_function=self.embeddings,
            persist_directory=DB_DIR,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print("[*] Tier 1 Correction Cache initialized.")

    def add_to_cache(self, user_query, correct_hs_code, duty_rate):
        """Used by Member D to inject manual corrections into the system."""
        self.cache_collection.add_texts(
            texts=[user_query],
            metadatas=[{"hs_code": correct_hs_code, "duty_rate": duty_rate, "is_verified": True}]
        )
        print(f"[+] Cached: {user_query} -> {correct_hs_code}")

    def check_cache(self, query):
        """Tier 1: Fast lookup. Returns None if no high-confidence match exists."""
        results = self.cache_collection.similarity_search_with_relevance_scores(query, k=1)
        if results and results[0][1] > 0.95: # 95% similarity threshold
            return results[0][0].metadata
        return None

if __name__ == "__main__":
    architect = TradeDataArchitect()
    architect.setup_correction_cache() 

    # SMART MODE: Only ingest if the database folder doesn't exist
    if not os.path.exists(DB_DIR):
        print("[*] Database not found. Starting first-time ingestion...")
        RAW_DATA_FOLDER = "./data_raw" 
        
        if os.path.exists(RAW_DATA_FOLDER):
            for filename in os.listdir(RAW_DATA_FOLDER):
                if filename.lower().endswith(".pdf"):
                    file_path = os.path.join(RAW_DATA_FOLDER, filename)
                    architect.ingest_document(file_path, country_code="IN", year=2026)
            print("[!] INITIAL INDEXING COMPLETE.")
        else:
            print(f"[!] ERROR: Put PDFs in {RAW_DATA_FOLDER} first!")
    else:
        # This is what will happen now since you already ran it once!
        print(f"[✓] Persistent Database found at {DB_DIR}. Entering Search Mode.")

    # Verification Test
    print("\n--- TEST SEARCH ---")
    results = architect.query_trade_law("lithium batteries", country_code="IN")
    if results:
        print(f"Success! Found relevant data in: {results[0].metadata['source']}")