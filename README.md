# 🔍 Smart Contract Analyzer

> API REST basée sur une architecture **RAG** (*Retrieval-Augmented Generation*) permettant d'interroger et d'analyser des smart contracts Solidity (OpenZeppelin) en langage naturel.

---

## 🚀 Démarrage Rapide

### 1. Installation

```bash
# Cloner le projet
git clone https://github.com/votre-utilisateur/smart-contract-analyzer.git
cd smart-contract-analyzer

# Créer et activer l'environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
.\venv\Scripts\Activate.ps1     # Windows (PowerShell)

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Configuration

Créez votre fichier `.env` et renseignez votre clé [Groq (gratuite)](https://console.groq.com/keys) :

```bash
cp .env.example .env            # Linux / macOS
Copy-Item .env.example .env    # Windows
```

```env
GROQ_API_KEY=gsk_votre_cle_ici
```

### 3. Lancer l'API

```bash
python main.py
# ou : uvicorn main:app --reload
```

L'API est accessible sur **http://127.0.0.1:8000** *(ChromaDB est initialisé automatiquement au premier démarrage)*.

---

## 🧪 Tests Automatisés

Le projet inclut une suite de **7 tests automatisés** validant l'ingestion, le vectorstore et l'API :

```bash
pytest -v
```

---

## 📡 Utilisation de l'API

* **Documentation interactive (Swagger UI)** : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Exemple de requête (`POST /ask`) :

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "Comment fonctionne la pause d urgence dans Pausable ?"}'
```

**Réponse JSON :**
```json
{
  "question": "Comment fonctionne la pause d urgence dans Pausable ?",
  "answer": "Le contrat Pausable.sol permet de bloquer temporairement des fonctions grâce au modificateur whenNotPaused...",
  "sources": ["Pausable.sol"]
}
```

---

## 🏗️ Stack Technique

* **Framework Web** : FastAPI, Uvicorn, Pydantic
* **Pipeline RAG** : LangChain (découpage ciblé pour le code Solidity : `function`, `modifier`)
* **Vectorstore & Embeddings** : ChromaDB + HuggingFace `all-MiniLM-L6-v2` (100% local sur CPU, gratuit)
* **Génération LLM** : Groq API (`openai/gpt-oss-20b`) pour une inférence ultra-rapide
* **Données indexées** : Contrats OpenZeppelin (`ERC20`, `ERC721`, `AccessControl`, `Ownable`, `Pausable`) dans `data/contracts/`
