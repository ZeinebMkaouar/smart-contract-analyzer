"""
Vectorstore : transforme les chunks en embeddings et les stocke dans ChromaDB.
"""

import os
import sys
from pathlib import Path

# Permet d'exécuter ce script directement (python src/vectorstore.py) ou comme module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Désactive les avertissements de symlinks HuggingFace sur Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME

# Singletons en mémoire pour éviter les rechargements lourds et les verrous SQLite
_embedding_model = None
_vectorstore = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Charge le modèle d'embedding local (gratuit, tourne sur CPU)."""
    global _embedding_model
    if _embedding_model is None:
        try:
            # Tente de charger depuis le cache local (évite les retries réseau quand hors-ligne)
            _embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu", "local_files_only": True},
            )
        except Exception:
            # Première exécution : téléchargement initial depuis HuggingFace
            _embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
            )
    return _embedding_model


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """
    Crée (ou écrase) la base vectorielle ChromaDB à partir des chunks fournis.
    Persiste automatiquement sur disque dans CHROMA_DB_DIR.
    """
    global _vectorstore
    embeddings = get_embedding_model()

    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DB_DIR),
    )

    print(f"✅ Base vectorielle créée avec {len(chunks)} chunks, sauvegardée dans {CHROMA_DB_DIR}")
    return _vectorstore


def load_vectorstore() -> Chroma:
    """
    Charge une base vectorielle déjà existante depuis le disque
    (évite de re-générer les embeddings à chaque lancement de l'API).
    """
    global _vectorstore
    if _vectorstore is None:
        embeddings = get_embedding_model()
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DB_DIR),
        )
    return _vectorstore


# Bloc de test : construit la base vectorielle à partir de zéro
if __name__ == "__main__":
    from src.ingestion import get_chunks

    chunks = get_chunks()
    vectorstore = build_vectorstore(chunks)

    print("\n--- Test rapide de recherche ---")
    results = vectorstore.similarity_search("Comment fonctionne la pause d'urgence ?", k=2)
    for i, doc in enumerate(results, 1):
        print(f"\nRésultat {i} (source: {doc.metadata['source']}) :")
        print(doc.page_content[:200])