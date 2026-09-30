"""
Retrieval : recherche sémantique dans la base vectorielle.
"""

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import TOP_K_RESULTS
from src.vectorstore import load_vectorstore


def retrieve_relevant_chunks(question: str, vectorstore: Chroma = None) -> list[Document]:
    """
    Retourne les chunks les plus pertinents pour une question donnée.
    Si aucun vectorstore n'est fourni, charge celui déjà existant sur disque.
    """
    if vectorstore is None:
        vectorstore = load_vectorstore()

    results = vectorstore.similarity_search(question, k=TOP_K_RESULTS)
    return results


def format_context(chunks: list[Document]) -> str:
    """
    Assemble les chunks récupérés en un seul bloc de texte,
    avec la source de chaque passage indiquée — utile pour que le LLM
    puisse citer d'où vient l'information dans sa réponse.
    """
    formatted_parts = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "inconnu")
        formatted_parts.append(f"[Source: {source}]\n{chunk.page_content}")

    return "\n\n---\n\n".join(formatted_parts)


# Bloc de test
if __name__ == "__main__":
    question = "Qui peut mint de nouveaux tokens dans un contrat ERC721 ?"

    chunks = retrieve_relevant_chunks(question)
    context = format_context(chunks)

    print(f"Question : {question}\n")
    print(f"--- Contexte assemblé ({len(chunks)} chunks) ---\n")
    print(context)