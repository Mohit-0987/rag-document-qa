import os
import tempfile

import streamlit as st

from ingest import load_and_chunk, build_vectorstore
from qa_chain import load_qa_chain, answer_question


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="centered"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📄 RAG Document Q&A Chatbot")

st.write(
    "Upload a PDF and ask questions about its contents "
    "using Retrieval-Augmented Generation (RAG)."
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------

if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    if st.button("Process PDF"):

        with st.spinner("Processing PDF..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                pdf_path = temp_file.name

            try:

                # Load PDF
                chunks = load_and_chunk(pdf_path)

                # Create vector store
                vectorstore = build_vectorstore(chunks)

                # Create retriever + Gemini
                retriever, llm = load_qa_chain(
                    vectorstore
                )

                # Store in session
                st.session_state.vectorstore = vectorstore
                st.session_state.retriever = retriever
                st.session_state.llm = llm

                # New document = new conversation
                st.session_state.chat_history = []

                st.session_state.uploaded_file_name = (
                    uploaded_file.name
                )

                st.success(
                    f"PDF processed successfully! "
                    f"{len(chunks)} chunks created."
                )

            finally:

                if os.path.exists(pdf_path):
                    os.remove(pdf_path)


# --------------------------------------------------
# CHAT AREA
# --------------------------------------------------

if st.session_state.retriever is not None:

    st.divider()

    st.subheader("💬 Ask a question")


    # --------------------------------------------------
    # DISPLAY PREVIOUS CHAT
    # --------------------------------------------------

    for message in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(message["question"])

        with st.chat_message("assistant"):
            st.write(message["answer"])


    # --------------------------------------------------
    # NEW QUESTION
    # --------------------------------------------------

    question = st.chat_input(
        "Ask a question about the document..."
    )


    if question:

        # Show user question immediately
        with st.chat_message("user"):
            st.write(question)


        # --------------------------------------------------
        # GENERATE ANSWER
        # --------------------------------------------------

        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "Searching the document..."
                ):

                    answer = answer_question(
                        question=question,
                        retriever=st.session_state.retriever,
                        llm=st.session_state.llm,
                        chat_history=st.session_state.chat_history
                    )

                st.write(answer)

            except Exception as e:

                error_message = str(e).lower()

                # ------------------------------------------
                # GEMINI QUOTA / RATE LIMIT ERROR
                # ------------------------------------------

                if (
                    "resource_exhausted" in error_message
                    or "quota" in error_message
                    or "429" in error_message
                    or "rate limit" in error_message
                ):

                    answer = (
                        "⚠️ **Daily AI usage limit reached**\n\n"
                        "The chatbot has reached its daily Gemini API "
                        "usage limit. Please try again later."
                    )

                    st.warning(answer)

                # ------------------------------------------
                # OTHER GEMINI/API ERRORS
                # ------------------------------------------

                else:

                    answer = (
                        "⚠️ **Unable to generate an answer**\n\n"
                        "The AI service encountered an error. "
                        "Please try again later."
                    )

                    st.error(answer)


        # --------------------------------------------------
        # SAVE CHAT
        # --------------------------------------------------

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer
            }
        )


    # --------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------

    if st.session_state.chat_history:

        if st.button("🗑️ Clear Chat"):

            st.session_state.chat_history = []

            st.rerun()