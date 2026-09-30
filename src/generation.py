"""
Generation : envoie le contexte récupéré + la question à Groq pour générer une réponse.
"""

import os
from dotenv import load_dotenv
from groq import Groq

from src.config import GROQ_MODEL_NAME

load_dotenv()  # charge les variables depuis le fichier .env

_client = None  # instance du client Groq, créée une seule fois (singleton simple)


def get_groq_client() -> Groq:
    """Retourne le client Groq, en le créant une seule fois."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY introuvable. Vérifie que ton fichier .env "
                "existe bien à la racine et contient GROQ_API_KEY=..."
            )
        _client = Groq(api_key=api_key)
    return _client


SYSTEM_PROMPT = """Tu es un assistant expert en smart contracts Solidity.
Réponds à la question de l'utilisateur en te basant UNIQUEMENT sur le contexte fourni ci-dessous.
Si le contexte ne contient pas assez d'information pour répondre, dis-le clairement au lieu d'inventer une réponse.
Cite le nom du fichier source quand c'est pertinent.
Réponds de façon claire et concise, en français."""


def generate_answer(question: str, context: str) -> str:
    """
    Génère une réponse en langage naturel à partir de la question
    et du contexte (chunks pertinents) récupéré par le retrieval.
    """
    client = get_groq_client()

    user_prompt = f"""Contexte extrait des smart contracts :

{context}

---

Question : {question}"""

    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # réponses factuelles et cohérentes, peu de "créativité"
    )

    return response.choices[0].message.content


# Bloc de test : pipeline complet retrieval + generation
if __name__ == "__main__":
    from src.retrieval import retrieve_relevant_chunks, format_context

    question = "Qui peut mint de nouveaux tokens dans un contrat ERC721 ?"

    chunks = retrieve_relevant_chunks(question)
    context = format_context(chunks)

    print(f"Question : {question}\n")
    print("Génération de la réponse avec Groq...\n")

    answer = generate_answer(question, context)
    print(f"--- Réponse ---\n{answer}")