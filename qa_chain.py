import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def load_qa_chain(vectorstore):

    # Retrieve the most relevant document chunks
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2
    )

    return retriever, llm


def answer_question(question, retriever, llm, chat_history=None):

    # --------------------------------------------------
    # STEP 1: Build conversation history
    # --------------------------------------------------

    history_text = ""

    if chat_history:

        for item in chat_history:
            history_text += (
                f"User: {item['question']}\n"
                f"Assistant: {item['answer']}\n"
            )

    # --------------------------------------------------
    # STEP 2: Retrieve relevant document chunks
    # --------------------------------------------------

    documents = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # --------------------------------------------------
    # STEP 3: Generate answer using Gemini
    # --------------------------------------------------

    prompt = f"""
You are a document question-answering assistant.

Your job is to answer the user's question using ONLY
the information contained in the document context.

Use the previous conversation only to understand
what the user is referring to.

Do NOT use outside knowledge.

If the answer cannot be found in the document context,
respond exactly:

I couldn't find that information in the document.

Keep the answer clear, concise, and directly related
to the user's question.

Previous conversation:
{history_text}

Document context:
{context}

User's question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    # --------------------------------------------------
    # STEP 4: Extract Gemini text
    # --------------------------------------------------

    if isinstance(response.content, list):

        answer = ""

        for item in response.content:

            if isinstance(item, dict) and item.get("type") == "text":
                answer += item.get("text", "")

        answer = answer.strip()

    else:
        answer = response.content.strip()

    return answer