import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def load_qa_chain(vectorstore):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2
    )

    return retriever, llm


def answer_question(question, retriever, llm, chat_history=None):

    # --------------------------------------------------
    # STEP 1: Rewrite follow-up question
    # --------------------------------------------------

    history_text = ""

    if chat_history:

        for item in chat_history:
            history_text += (
                f"User: {item['question']}\n"
                f"Assistant: {item['answer']}\n"
            )

    rewrite_prompt = f"""
You are helping a document question-answering system.

Convert the user's current question into a standalone question
that can be understood without the previous conversation.

Previous conversation:
{history_text}

Current question:
{question}

Rules:
- If the question is already standalone, return it unchanged.
- If it refers to something from the previous conversation,
  include that subject in the rewritten question.
- Do not answer the question.
- Return ONLY the rewritten question.
"""

    rewritten_response = llm.invoke(rewrite_prompt)

    if isinstance(rewritten_response.content, list):

        standalone_question = ""

        for item in rewritten_response.content:
            if isinstance(item, dict) and item.get("type") == "text":
                standalone_question += item.get("text", "")

        standalone_question = standalone_question.strip()

    else:
        standalone_question = rewritten_response.content.strip()


    # --------------------------------------------------
    # STEP 2: Retrieve document chunks
    # --------------------------------------------------

    documents = retriever.invoke(standalone_question)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )


    # --------------------------------------------------
    # STEP 3: Generate answer
    # --------------------------------------------------

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
contained in the document context.

You may use the conversation history to understand
what the user is referring to.

Do NOT use outside knowledge.

If the answer cannot be found in the document context,
respond exactly:

I couldn't find that information in the document.

Keep the answer clear and concise.

Previous conversation:
{history_text}

Document context:
{context}

User's original question:
{question}

Standalone question used for retrieval:
{standalone_question}

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
        answer = response.content


    return answer