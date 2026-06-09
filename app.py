import streamlit as st
from query import query

st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="🤔",
    layout="wide"
)

st.title("Local RAG Assistant")
st.caption("Ask questions about the documents you added - powered by Ollama and ChromaDB")

#Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

#Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#Chat
#takes user inputs and sends out output
if user_input := st.chat_input("Ask something about your documents..."):

    #Shower user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    #Run RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer, sources = query(user_input)

        #Display answer
        st.markdown(answer)

        #Displays sources
        if sources:
            with st.expander("sources"):
                for source in set(sources):
                    st.markdown(f"- '{source}'")

    st.session_state.messages.append({"role": "assistant", "content": answer})