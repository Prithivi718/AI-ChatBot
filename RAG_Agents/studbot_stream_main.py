import streamlit as st
from pathlib import Path
import speech_recognition as sr
import pyttsx3
import sys


# Temporarily remove torch from sys.modules during Streamlit module watching
sys.modules['torch.classes'] = None

from studbot_upload import upload_document
#from studbot_retrieve import retrieve_and_respond
from checkexist_docs import check_existing_collection


# ---------------------- Streamlit Layout ----------------------
st.set_page_config(page_title="Student Assistant Bot", layout="centered")
st.title("🎓 Student Assistant Bot")
st.caption("Ask questions based on your uploaded document 📄")

# ---------------------- Session State ----------------------
if "collection" not in st.session_state:
    st.session_state.collection = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------- Text-to-Speech Engine ----------------------
engine = pyttsx3.init()
engine.setProperty("voice", engine.getProperty('voices')[0].id)
engine.setProperty("rate", 170)
engine.setProperty("volume", 0.9)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ---------------------- Speech Recognition ----------------------
recognizer = sr.Recognizer()

def listen_commands():
    with sr.Microphone() as source:
        st.info("🎙️ Listening...")
        recognizer.pause_threshold = 1
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=10, phrase_time_limit=6)

    try:
        st.info("🧠 Recognizing...")
        text = recognizer.recognize_google(audio, language='en-US')
        return text.lower()
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that.")
        return None
    except sr.RequestError as e:
        speak(f"Speech recognition error: {e}")
        return None
    except Exception as ex:
        st.error(f"❌ Error: {ex}")
        return None

# ---------------------- Sidebar Upload ----------------------
with st.sidebar:
    file_path = st.chat_input("📁 Enter the full path of your document:")
    if file_path:
        file_path = str(Path(file_path.strip()))
        if not st.session_state.collection:
            st.info("📚 Checking document in Qdrant...")
            st.session_state.collection = check_existing_collection(file_path)
            if not st.session_state.collection:
                st.info("📤 Uploading document and creating collection...")
                file_path = file_path.replace("\\", "\\\\")  # Safety
                st.session_state.collection = upload_document(file_path)
                # st.success("✅ Document processed successfully!")

#---------------------- API Requests ------------------------
import requests
API_URL = "http://localhost:8000"
# ---------------------- Voice Input and Chat ----------------------
try:

    if st.session_state.collection:
        if st.button("🎤 Ask with Voice"):
            question = listen_commands()
            if question:
                # Prepare the Data for the Request
                data = {
                    "collection": st.session_state.collection,
                    "question": question
                }
                with st.chat_message("user"):
                    st.markdown(f"🗣️ **You:** {question}")
                st.session_state.chat_history.append(("user", question))

                #Request call
                response= requests.post(f"{API_URL}/chat", json=data)

                #response = retrieve_and_respond(st.session_state.collection, question)
                with st.chat_message("assistant"):
                    st.markdown("💬 Generating answer...")
                    if response.status_code == 200:
                        response_text= response.json()
                        st.markdown(response_text)
                        st.session_state.chat_history.append(("assistant", response))
                    else:
                        st.error("Error getting response from mcp")
except Exception as e:
    st.error(f"Error: {str(e)}")
# ---------------------- Chat History Display ----------------------
st.divider()
st.subheader("🕘 Conversation History")
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
