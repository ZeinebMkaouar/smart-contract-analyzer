"""
API REST exposant le pipeline RAG Smart Contract Analyzer.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import CHROMA_DB_DIR
from src.ingestion import get_chunks
from src.vectorstore import build_vectorstore, load_vectorstore
from src.retrieval import retrieve_relevant_chunks, format_context
from src.generation import generate_answer

# Stocke le vectorstore chargé en mémoire, pour ne pas le recharger à chaque requête
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Exécuté une seule fois au démarrage de l'API.
    Si la base vectorielle n'existe pas encore sur le disque (ex: premier déploiement
    sur Render, où chroma_db/ n'est pas versionné sur Git), on la construit automatiquement
    à partir des contrats sources. Sinon, on charge simplement la base existante.
    """
    if not CHROMA_DB_DIR.exists() or not any(CHROMA_DB_DIR.iterdir()):
        print("⚠️  Aucune base vectorielle trouvée — construction en cours...")
        chunks = get_chunks()
        vectorstore = build_vectorstore(chunks)
    else:
        print("✅ Base vectorielle existante détectée — chargement...")
        vectorstore = load_vectorstore()

    app_state["vectorstore"] = vectorstore
    print("🚀 API prête à recevoir des requêtes.")

    yield  # l'API tourne normalement ici

    app_state.clear()  # nettoyage à l'arrêt de l'API


app = FastAPI(
    title="Smart Contract Analyzer API",
    description="API RAG pour analyser et interroger des smart contracts en langage naturel",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionInput(BaseModel):
    question: str


class AnswerOutput(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API Smart Contract Analyzer 🔍"}


@app.post("/ask", response_model=AnswerOutput)
def ask_question(data: QuestionInput):
    """
    Reçoit une question en langage naturel, effectue le retrieval,
    puis génère une réponse basée sur les smart contracts indexés.
    """
    if not data.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    vectorstore = app_state["vectorstore"]

    chunks = retrieve_relevant_chunks(data.question, vectorstore=vectorstore)
    context = format_context(chunks)
    answer = generate_answer(data.question, context)

    sources = sorted(set(chunk.metadata.get("source", "inconnu") for chunk in chunks))

    return AnswerOutput(
        question=data.question,
        answer=answer,
        sources=sources,
    )