"""
Vectorstore : transforme les chunks en embeddings et les stocke dans ChromaDB.
"""

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Charge le modèle d'embedding local (gratuit, tourne sur CPU)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """
    Crée (ou écrase) la base vectorielle ChromaDB à partir des chunks fournis.
    Persiste automatiquement sur disque dans CHROMA_DB_DIR.
    """
    embeddings = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DB_DIR),
    )

    print(f"✅ Base vectorielle créée avec {len(chunks)} chunks, sauvegardée dans {CHROMA_DB_DIR}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """
    Charge une base vectorielle déjà existante depuis le disque
    (évite de re-générer les embeddings à chaque lancement de l'API).
    """
    embeddings = get_embedding_model()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )
    return vectorstore


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