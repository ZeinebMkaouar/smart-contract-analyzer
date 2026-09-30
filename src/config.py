"""
Configuration centralisée du projet Smart Contract Analyzer.
Toutes les constantes partagées (chemins, noms de modèles...) vivent ici.
"""

from pathlib import Path

# --- Chemins ---
BASE_DIR = Path(__file__).resolve().parent.parent  # racine du projet
CONTRACTS_DIR = BASE_DIR / "data" / "contracts"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# --- Chunking ---
CHUNK_SIZE = 1000        # nombre de caractères par chunk
CHUNK_OVERLAP = 150      # chevauchement entre chunks consécutifs (évite de couper une idée en deux)

# --- Embeddings ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # modèle sentence-transformers, léger et rapide

# --- Base vectorielle ---
COLLECTION_NAME = "smart_contracts"

# --- LLM (Groq) ---
GROQ_MODEL_NAME = "openai/gpt-oss-20b"   # rapide, bon pour dev/test
# Alternative pour de meilleures réponses (plus lent) : "openai/gpt-oss-120b"

# --- Retrieval ---
TOP_K_RESULTS = 4   # nombre de chunks les plus pertinents à récupérer par question