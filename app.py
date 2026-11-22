import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/ask"
)
import streamlit as st
import requests

st.set_page_config(
    page_title="Spotify Architecture RAG",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Spotify Architecture AI Assistant")
st.write("Ask questions about the Spotify Web Application Architecture PDF.")



# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
question = st.chat_input("Ask a question about the PDF...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching document and generating answer..."):

            try:
                response = requests.post(
                    API_URL,
                    params={"question": question}
                )

                response.raise_for_status()

                data = response.json()

                answer = data["answer"]

                st.markdown(answer)

                # Display sources
                with st.expander("📚 View Retrieved Sources"):

                    for i, source in enumerate(data["sources"], 1):

                        st.markdown(f"### Source {i}")

                        st.write("Page:", source["page"])

                        st.write("File:", source["source"])

                        st.write(source["text"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Error connecting to backend: {e}")