"""
Tests unitaires et d'intégration pour Smart Contract Analyzer.
"""

import os
from unittest.mock import patch
import pytest
from fastapi import HTTPException

# Force le mode local et silencieux pour les tests
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from main import read_root, ask_question, QuestionInput, app_state
from src.ingestion import load_contracts, split_documents
from src.vectorstore import load_vectorstore
from src.retrieval import retrieve_relevant_chunks, format_context


# --- Tests Ingestion ---

def test_load_contracts():
    """Vérifie que les 5 contrats OpenZeppelin sont bien chargés."""
    docs = load_contracts()
    assert len(docs) == 5
    sources = [doc.metadata["source"] for doc in docs]
    assert "ERC20.sol" in sources
    assert "ERC721.sol" in sources
    assert "Pausable.sol" in sources
    assert "Ownable.sol" in sources
    assert "AccessControl.sol" in sources


def test_split_documents():
    """Vérifie que le découpage en chunks produit des segments non vides avec métadonnées."""
    docs = load_contracts()
    chunks = split_documents(docs)
    assert len(chunks) > len(docs)
    for chunk in chunks:
        assert len(chunk.page_content.strip()) > 0
        assert "source" in chunk.metadata


# --- Tests Retrieval ---

def test_retrieval_and_context():
    """Vérifie que la recherche sémantique renvoie bien les chunks pertinents avec leur source."""
    vectorstore = load_vectorstore()
    question = "Comment fonctionne la pause d'urgence ?"
    chunks = retrieve_relevant_chunks(question, vectorstore=vectorstore)

    assert len(chunks) > 0
    sources = [c.metadata.get("source") for c in chunks]
    assert "Pausable.sol" in sources

    context = format_context(chunks)
    assert "[Source: Pausable.sol]" in context


# --- Tests FastAPI Routes ---

def test_read_root():
    """Vérifie l'endpoint de base GET /."""
    response = read_root()
    assert "Smart Contract Analyzer" in response["message"]


def test_ask_empty_question():
    """Vérifie que l'API rejette les questions vides avec un code 400."""
    with pytest.raises(HTTPException) as exc_info:
        ask_question(QuestionInput(question="   "))
    assert exc_info.value.status_code == 400
    assert "vide" in exc_info.value.detail.lower()


def test_ask_question_missing_api_key():
    """Vérifie qu'un appel sans clé GROQ_API_KEY retourne un code 500 explicite."""
    app_state["vectorstore"] = load_vectorstore()
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(HTTPException) as exc_info:
            ask_question(QuestionInput(question="Comment mettre en pause le contrat ?"))
        assert exc_info.value.status_code in (500, 502)
        assert "GROQ_API_KEY" in exc_info.value.detail or "groq" in exc_info.value.detail.lower()


def test_ask_question_mocked_llm():
    """Vérifie le pipeline complet /ask de la requête au retour structuré."""
    app_state["vectorstore"] = load_vectorstore()
    with patch("main.generate_answer", return_value="Le contrat peut être mis en pause via _pause() par un compte autorisé."):
        result = ask_question(QuestionInput(question="Comment mettre en pause le contrat ?"))
        assert result.question == "Comment mettre en pause le contrat ?"
        assert "mis en pause" in result.answer
        assert "Pausable.sol" in result.sources
