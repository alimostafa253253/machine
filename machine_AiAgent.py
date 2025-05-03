from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv
import os
import pyttsx3
import speech_recognition as sr


# UI Setup
st.title("🎙️ Speech Translator")

input_lang_options = [
    "ar-EG",  # Arabic
    "en-US",  # English (US)
    "es-ES",  # Spanish
    "fr-FR",  # French
    "de-DE",  # German
    "it-IT"   # Italian
]

output_lang_options = [
    "en",     # English
    "es",     # Spanish
    "fr",     # French
    "ar",     # Arabic
    "de",     # German
    "it",     # Italian
    "pt"      # Portuguese (kept as important global language)
]

# UI Elements
input_lang = st.selectbox("🎤 Speak in:", input_lang_options, index=0)
output_lang = st.selectbox("🌍 Translate to:", output_lang_options, index=0)
translation_type = st.radio("🔊 Output:", [ "Text + Speech"], horizontal=True)
speech_time = st.slider("⏱️ Recording Time (seconds):", min_value=3, max_value=10, value=7)

# Initialize Components
recognizer = sr.Recognizer()


def create_speech_engine(language="en"):
    engine = pyttsx3.init()
    engine.setProperty('rate', 145)
    engine.setProperty('volume', 0.9)
    
    # Set voice language based on the output language
    voices = engine.getProperty('voices')
    if language == "ar":
        engine.setProperty('voice', voices[1].id)  # Arabic voice (ensure it's available on your system)
    elif language == "es":
        engine.setProperty('voice', voices[2].id)  # Spanish voice (ensure it's available on your system)
    else:
        engine.setProperty('voice', voices[0].id)  # Default voice (English)
    
    return engine

def speak(text, language="en"):
    try:
        engine = create_speech_engine(language)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        st.error("Audio playback failed")

load_dotenv()
# langsmith tracking
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_TRACING_V2'] = 'true'
os.environ['LANGSMITH_PROJECT'] = 'Machine_Translation'

# Sidebar - User enters Groq API Key
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

# Sidebar - Select the Groq AI Model
engine = st.sidebar.selectbox("Select Groq AI Model", ['llama3-70b-8192', 'Deepseek-R1-Distill-Llama-70b'])

# Sidebar - Adjust Response Parameters
temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider('Max Tokens', min_value=50, max_value=300, value=150)

# Define the prompt
system_template = "Translate the following text into {language}: {sentence}"
prompt = ChatPromptTemplate.from_template(system_template)

# Response Function
def generate_response(sentence, language, api_key, engine, temperature, max_tokens):
    if not api_key:
        return "⚠️ Please enter your Groq API Key in the sidebar."

    llm = ChatGroq(model=engine, groq_api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    
    # Chain Execution
    chain = prompt | llm | output_parser
    answer = chain.invoke({"language": language, "sentence": sentence})
    
    return answer

# Speech Recognition Function
def record_speech():
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            st.info("🔴 Listening for speech...")
            audio = recognizer.listen(source, timeout=speech_time)
            st.info("🔄 Recognizing speech...")
            text = recognizer.recognize_google(audio, language=input_lang)
            st.write(f"🎤 You said: {text}")
            return text
    except sr.UnknownValueError:
        st.error("👂 Could not understand the audio. Please try again.")
        return None
    except sr.RequestError as e:
        st.error(f"⚠️ Speech recognition error: {e}")
        return None

# Initialize user_input variable
user_input = None

# Main interface for user input
st.write("Enter a sentence to translate:")

# Button to start speech recognition
if st.button("🎤 Start Speech Recognition"):
    user_input = record_speech()

if user_input:
    response = generate_response(user_input, output_lang, api_key, engine, temperature, max_tokens)
    st.write(f"**Translation in {output_lang}:**")
    st.write(response)
    
    if "Text + Speech" in translation_type:
        speak(response, language=output_lang)
else:
    st.write("Please provide a sentence to translate.")
