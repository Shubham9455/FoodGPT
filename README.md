# 🍽️ FoodGPT — AI-Powered Restaurant Recommendation System

AI-powered restaurant discovery system that understands natural language queries and returns personalized recommendations using a Retrieval-Augmented Generation (RAG) pipeline.

---


## 🧠 How It Works

<img width="765" height="556" alt="image" src="https://github.com/user-attachments/assets/7dcae22a-1412-462a-9cb5-c06a45d2e6e1" />

---

## ✨ Features

* 🔍 Natural language search (e.g. *"cheap chicken under ₹300"*)
* ⚡ Semantic search using FAISS + sentence-transformers
* 🤖 LLM-powered responses using Ollama (Mistral / LLaMA3)
* 🔄 Real-time streaming responses (SSE)
* 💬 ChatGPT-style UI (Next.js)
* 🍽️ Structured restaurant cards (name, rating, cuisine, location)
* 🧠 Prompt grounding to prevent hallucinations

---

## 🏗️ Tech Stack

**Backend**

* FastAPI
* FAISS
* Sentence Transformers
* Ollama (Local LLM)

**Frontend**

* Next.js (App Router)
* Tailwind CSS

---

## 📂 Project Structure

```
foodgpt/
│── backend/
│   │── main.py
│   │── rag_engine.py
│   │── embeddings/
│   │── data/
│   │── build_index.py
│
│── frontend/
│   │── app/
```

---

## ⚙️ Setup Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/foodgpt.git
cd foodgpt
```

---

### 2️⃣ Backend Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

### 3️⃣ Install Ollama (LLM)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Run:

```bash
ollama pull mistral
ollama serve
```

---

### 4️⃣ Build FAISS Index (one-time)

```bash
cd ..
python build_index.py
```

---

### 5️⃣ Start Backend

```bash
cd backend
fastapi dev main.py
```

Backend runs on:

```
http://127.0.0.1:8000
```

---

### 6️⃣ Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on:

```
http://localhost:3000
```

---

## 🧠 Architecture

User Query → Embedding → FAISS Search → Context Injection → LLM → Streaming Response

---

## 📸 Screenshots

### Chat Interface

![Chat](./assets/chat.png)

### Recommendations Panel

![Cards](./assets/cards.png)

---

## 🚧 Future Improvements

* 🗺️ Map integration
* ⭐ Rating filters
* ❤️ Favorites system
* 📊 Sorting & personalization

---

## 💡 Key Learnings

* Built a complete RAG pipeline from scratch
* Implemented real-time streaming with SSE
* Controlled LLM hallucination via prompt grounding
* Designed full-stack AI product with real UX

---

## ⭐ Star this repo if you like it!
