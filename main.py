"""
API REST exposant le pipeline RAG Smart Contract Analyzer.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.retrieval import retrieve_relevant_chunks, format_context
from src.generation import generate_answer

app = FastAPI(
    title="Smart Contract Analyzer API",
    description="API RAG pour analyser et interroger des smart contracts en langage naturel",
    version="1.0.0",
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

    chunks = retrieve_relevant_chunks(data.question)
    context = format_context(chunks)
    answer = generate_answer(data.question, context)

    sources = sorted(set(chunk.metadata.get("source", "inconnu") for chunk in chunks))

    return AnswerOutput(
        question=data.question,
        answer=answer,
        sources=sources,
    )