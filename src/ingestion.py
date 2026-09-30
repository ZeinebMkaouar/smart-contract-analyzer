"""
Ingestion : chargement des smart contracts et découpage en chunks.
"""

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import CONTRACTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_contracts() -> list[Document]:
    """
    Charge tous les fichiers .sol du dossier data/contracts/
    et les transforme en objets Document LangChain,
    avec le nom du fichier source conservé en métadonnée.
    """
    documents = []

    sol_files = sorted(Path(CONTRACTS_DIR).glob("*.sol"))

    if not sol_files:
        raise FileNotFoundError(
            f"Aucun fichier .sol trouvé dans {CONTRACTS_DIR}. "
            "Vérifie que tes contrats sont bien à cet emplacement."
        )

    for file_path in sol_files:
        content = file_path.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=content,
                metadata={"source": file_path.name}
            )
        )

    print(f"✅ {len(documents)} contrat(s) chargé(s) : {[d.metadata['source'] for d in documents]}")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Découpe les documents en chunks, en essayant de respecter
    la structure du code Solidity (fonctions, accolades) autant que possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n    function ", "\n    modifier ", "\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    print(f"✅ {len(documents)} document(s) découpé(s) en {len(chunks)} chunks.")
    return chunks


def get_chunks() -> list[Document]:
    """Point d'entrée principal : charge et découpe les contrats."""
    documents = load_contracts()
    chunks = split_documents(documents)
    return chunks


# Bloc de test rapide — exécute ce fichier directement pour vérifier que tout fonctionne
if __name__ == "__main__":
    chunks = get_chunks()
    print("\n--- Aperçu du premier chunk ---")
    print(f"Source : {chunks[0].metadata['source']}")
    print(chunks[0].page_content[:300])
    print("...")