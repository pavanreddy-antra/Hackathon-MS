import os

import streamlit as st
from hackathon.csp import AzureCSP
from audiorecorder import audiorecorder

MESSAGE_HISTORY_LENGTH = 20


class Server:
    def __init__(self):
        self.csp = AzureCSP(embeddings='azure', chat='azure', stt='azure')

    def main(self):
        st.set_page_config(page_title="Chat-Abby", page_icon=":teacher:", layout="wide")
        st.image("tw.png", width=200)
        st.title("Microsoft AI Hackathon Submission")
        st.header("Talk to Abby, Your Personal Addiction Issues Counselor")

        if 'transcription' not in st.session_state:
            st.session_state.transcription = ""
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'state' not in st.session_state:
            st.session_state.state = "started"
        if 'copy_to_input' not in st.session_state:
            st.session_state.copy_to_input = False

        col1, col2, col3 = st.columns([8, 1, 1])

        with col1:
            user_question = st.text_input(
                "Type your question and press enter:",
                value=st.session_state.transcription if st.session_state.copy_to_input else "",
                key="question"
            )

        with col2:
            audio = audiorecorder("Click to record", "Click to stop recording")
            if len(audio) > 0:
                st.audio(audio.export().read())
                audio.export("audio.wav", format="wav")

                text = self.csp.stt_client.get_text("audio.wav")
                st.session_state.transcription = text
                st.session_state.copy_to_input = True

        with col3:
            if st.button("Copy Transcription to Input"):
                st.session_state.copy_to_input = False  # Reset the flag after copying
                st.session_state.transcription = user_question



        if st.button("Send"):
            if user_question:
                st.session_state.conversation.append(f"You: {user_question}\n")

                if len(st.session_state.history) >= MESSAGE_HISTORY_LENGTH:
                    active_history = st.session_state.history[-MESSAGE_HISTORY_LENGTH:]
                    st.session_state.history = st.session_state.history[:-MESSAGE_HISTORY_LENGTH]
                else:
                    active_history = st.session_state.history.copy()
                    st.session_state.history = []

                response, updated_history, updated_state = self.csp.start_conversation(user_question, active_history, st.session_state.state)
                st.session_state.conversation.append(f"Abby: {response}\n")

                st.session_state.history += updated_history
                st.session_state.state = updated_state

        for i in range(0, len(st.session_state.conversation), 2):
            st.write(list(reversed(st.session_state.conversation))[i + 1])
            st.write(list(reversed(st.session_state.conversation))[i])
            st.write("_" * 100)


if __name__ == '__main__':
    Server().main()
