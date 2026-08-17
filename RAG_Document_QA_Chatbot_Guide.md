# RAG-Based Document Q&A Chatbot — Complete Build Guide

## 0. Is this simple or difficult?

**Simple-to-moderate.** You already know Python, and that's 80% of what this needs. There's no deep math, no model training, no GPU required. You're mainly **connecting existing tools together**: a document loader → a chunker → an embedding model → a vector database → an LLM API → a UI. Realistic timeline for you: **6-10 hours spread over 2-3 days**, including debugging.

The only genuinely new concepts are RAG, embeddings, and vector search — all explained below before you touch code.

---

## 1. What is this project?

A **RAG (Retrieval-Augmented Generation) chatbot** lets a user upload their own documents (PDFs, text files) and ask questions about them in plain English. Instead of the LLM answering from its general training knowledge (which can be outdated, generic, or wrong for your specific document), it first **retrieves the most relevant chunks of your document** and then **generates an answer grounded in that retrieved text**.

In short: it's "chat with your own PDF."

## 2. Why does this matter / why it helps you

- **It's the #1 practical GenAI pattern companies are hiring for right now.** Almost every "AI feature" shipped in products today (customer support bots, internal knowledge search, legal document review, HR policy bots) is RAG under the hood.
- It proves you can work with the modern AI stack: **LLM APIs, embeddings, vector databases, LangChain/orchestration** — skills your current resume didn't show at all.
- It's cheap and fast to build (no GPU, no training), unlike your other two projects which are compute-heavy — a good contrast to show range.
- It directly supports interview conversations: recruiters ask "have you worked with LLMs / GenAI?" — now you can say yes, in detail.

## 3. Real-world use cases (good to mention in interviews)

| Use case | Example |
|---|---|
| Customer support | Bot answers from product manuals/FAQs instead of a human agent |
| Internal knowledge base | Employees ask HR policy or IT-helpdesk questions |
| Legal / compliance | Lawyers query contracts, get answers with citations |
| Healthcare | Doctors query medical literature/patient guidelines |
| Education | Students query textbooks/lecture notes |
| Finance | Analysts query earnings reports, 10-Ks |

---

## 4. Core theory (understand this before coding)

### 4.1 Why not just paste the document into the LLM prompt?
LLMs have a limited **context window** (how much text they can read at once), and pasting huge documents is expensive and often exceeds that limit. RAG solves this by only sending the LLM the *relevant* few paragraphs, not the whole document.

### 4.2 What is an embedding?
An embedding model converts text into a list of numbers (a **vector**) that captures its *meaning*. Sentences with similar meaning end up with vectors that are close together in this numeric space, even if they use different words.

### 4.3 What is a vector database?
A database optimized for storing these vectors and quickly finding the ones most similar to a query vector (**similarity search**), instead of exact keyword matching. FAISS (by Meta) is a free, local, no-server-needed library that does this well for small/medium projects.

### 4.4 The RAG pipeline, end to end
1. **Load** the document (PDF/text).
2. **Chunk** it into smaller overlapping pieces (e.g., 500 words each) — because embeddings work better on focused text, not entire documents.
3. **Embed** each chunk into a vector, store all vectors in FAISS.
4. When the user asks a question, **embed the question** too.
5. **Retrieve** the top-k most similar chunks from FAISS (semantic search).
6. **Construct a prompt**: "Using only the following context, answer the question: [retrieved chunks] + [question]".
7. **Send to the LLM** (OpenAI/Gemini) → get the grounded answer.
8. **Return** the answer to the user via a UI.

That's the entire project. Everything below is just implementing these 8 steps.

---

## 5. Tech stack

| Component | Tool | Why |
|---|---|---|
| Language | Python | You already know it |
| Orchestration | LangChain | Glues the pipeline steps together |
| Embeddings | `sentence-transformers` (free, local) or Gemini/OpenAI embeddings | Free option avoids API cost while building |
| Vector DB | FAISS | Free, local, simple |
| LLM | Google Gemini API (**free tier available**) or OpenAI (paid) | Gemini recommended for a student budget |
| UI | Gradio | Fastest way to get a usable web UI in a few lines |
| Version control | Git + GitHub | You already use this |

> **Recommendation:** use the **Gemini API free tier** (`gemini-1.5-flash` or newer) — no credit card needed for the free quota, unlike OpenAI.

---

## 6. Step-by-step implementation (from zero)

### Step 1 — Set up your environment
```bash
mkdir rag-document-qa
cd rag-document-qa
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install langchain langchain-community langchain-google-genai
pip install faiss-cpu sentence-transformers pypdf gradio python-dotenv
```

### Step 2 — Get a free Gemini API key
1. Go to Google AI Studio (aistudio.google.com).
2. Sign in → "Get API key" → create a new key.
3. In your project folder, create a file named `.env`:
```
GOOGLE_API_KEY=your_key_here
```
4. Never commit `.env` to GitHub — add it to `.gitignore`.

### Step 3 — Load and chunk the document
Create `ingest.py`:
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # characters per chunk
        chunk_overlap=100    # overlap so context isn't cut mid-sentence
    )
    chunks = splitter.split_documents(pages)
    return chunks
```
*Why overlap?* If a sentence gets cut between two chunks, overlap ensures each chunk still has enough surrounding context to be understandable on its own.

### Step 4 — Embed the chunks and build the FAISS index
```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore
```
`all-MiniLM-L6-v2` is a small, free, fast embedding model that runs on CPU — good enough for this project and won't cost you anything.

### Step 5 — Set up retrieval + the LLM
Create `qa_chain.py`:
```python
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

load_dotenv()

def load_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # top 4 chunks

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2   # low temperature = more factual, less "creative"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain
```
*Why `temperature=0.2`?* For a Q&A system you want factual, grounded answers, not creative ones — a low temperature reduces randomness.

### Step 6 — Build the Gradio UI
Create `app.py`:
```python
import gradio as gr
from ingest import load_and_chunk
from ingest import build_vectorstore
from qa_chain import load_qa_chain

qa_chain = None

def process_pdf(pdf_file):
    global qa_chain
    chunks = load_and_chunk(pdf_file.name)
    build_vectorstore(chunks)
    qa_chain = load_qa_chain()
    return "Document processed. You can ask questions now."

def answer_question(question):
    if qa_chain is None:
        return "Please upload a document first."
    result = qa_chain.invoke({"query": question})
    return result["result"]

with gr.Blocks() as demo:
    gr.Markdown("# 📄 RAG Document Q&A Chatbot")
    pdf_input = gr.File(label="Upload PDF")
    upload_btn = gr.Button("Process Document")
    status = gr.Textbox(label="Status")
    upload_btn.click(process_pdf, inputs=pdf_input, outputs=status)

    question = gr.Textbox(label="Ask a question about the document")
    answer = gr.Textbox(label="Answer")
    ask_btn = gr.Button("Ask")
    ask_btn.click(answer_question, inputs=question, outputs=answer)

demo.launch()
```

### Step 7 — Run it
```bash
python app.py
```
Gradio will give you a local URL (e.g., `http://127.0.0.1:7860`). Open it, upload a PDF, click "Process Document," then ask questions.

### Step 8 — Test it properly
- Upload a PDF you know well (e.g., your own resume, a college syllabus, a research paper).
- Ask a question whose answer is clearly *in* the document — check it's correct.
- Ask a question *not* in the document — check the bot doesn't hallucinate a confident wrong answer (this is a known RAG weakness; you can mention it in interviews).

### Step 9 — Push to GitHub
```bash
git init
echo "venv/
.env
faiss_index/
__pycache__/" > .gitignore
git add .
git commit -m "Initial RAG document Q&A chatbot"
git branch -M main
git remote add origin https://github.com/<your-username>/rag-document-qa.git
git push -u origin main
```
Add a short `README.md` explaining what the project does, the tech stack, and how to run it — this is what recruiters actually open first.

### Step 10 (optional, makes it stronger)
- Support multiple file uploads, not just one.
- Show the **source chunk** used for each answer (`result["source_documents"]`) — builds trust and is a common real-world RAG feature.
- Deploy it for free on Hugging Face Spaces so you have a live demo link, not just code.

---

## 7. How this maps to your resume bullets

- *"Built a RAG chatbot using LangChain, FAISS, and an LLM API"* → Steps 3-6.
- *"Implemented document chunking, embedding generation, and semantic similarity search"* → Steps 3-4.
- *"Deployed an interactive Gradio interface... used Git for version control"* → Steps 6, 9.

Make sure what you actually build matches what's written — if you add features, feel free to make the bullets slightly more specific afterward.

---

## 8. Interview / theory questions on this project

**Conceptual**
1. What is RAG, and why is it used instead of just fine-tuning an LLM on your data?
2. What is an embedding, and how does semantic search differ from keyword search?
3. Why do we chunk documents before embedding them? What happens if chunks are too big or too small?
4. What is chunk overlap and why does it matter?
5. What does "top-k retrieval" mean, and how do you choose a good value of k?
6. What is a vector database, and how is FAISS different from a database like MySQL or PostgreSQL?
7. What does "temperature" control in an LLM API call?
8. What is hallucination in LLMs, and how does RAG reduce (but not eliminate) it?
9. What's the difference between RAG and prompt engineering alone?
10. How would you evaluate whether your RAG system's answers are actually good?

**Technical / implementation**
11. Walk me through your RAG pipeline end to end.
12. Why did you choose FAISS over other vector DBs like Pinecone or ChromaDB?
13. Why use a local embedding model (`all-MiniLM-L6-v2`) instead of an API-based one?
14. What would you change to support multiple documents instead of one?
15. How would you handle a very large document (e.g., a 500-page PDF) — what bottlenecks would you hit?
16. What happens if the user asks a question completely unrelated to the document?
17. How would you add memory so the bot supports follow-up questions (multi-turn conversation)?
18. How would you scale this from a local Gradio app to a production system serving many users?
19. What are the cost/latency trade-offs of using an LLM API vs. self-hosting an open-source LLM?
20. What security/privacy concerns exist if this were used with confidential company documents?

Be ready to answer #16 and #20 honestly — interviewers like when you can name the limitations of your own project (shows maturity), not just the strengths.
