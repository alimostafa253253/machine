import streamlit as st
import pyttsx3
import speech_recognition as sr
from googletrans import Translator
import threading
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import wandb
import os

# UI Setup
st.title("🎙️ Speech Translator")

# Language options
input_lang_options = ["ar-EG", "en-US" ]
output_lang_options = ["en",  "ar" ]

# UI Elements
input_lang = st.selectbox("🎤 Speak in:", input_lang_options, index=0)
output_lang = st.multiselect("🌍 Translate to:", output_lang_options, default=["en", "ar"])
translation_type = st.radio("🔊 Output:", [ "Text + Speech"], horizontal=True)
speech_time = st.slider("⏱️ Recording Time (seconds):", min_value=3, max_value=10, value=7)

# Initialize Components
recognizer = sr.Recognizer()
translator = Translator()

def create_speech_engine():
    engine = pyttsx3.init()
    engine.setProperty('rate', 145)
    engine.setProperty('volume', 0.9)
    return engine

def speak(text):
    try:
        engine = create_speech_engine()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        st.error("Audio playback failed")

# Initialize WandB
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_API_KEY"] = "4cf3591f262cd568777e73fcda947286ee03b410"
wandb.login()

artifact_paths = {
    "en_to_ar": "abdelaziz67-ain-shams-university/egyptian-arabic-translation-finetuning/translation_model:v10",
    "ar_to_en": "abdelaziz67-ain-shams-university/egyptian-english-translation-finetuning/translation_model:v1",
}

def load_model(artifact_path, local_path):
    if os.path.exists(local_path):
        return local_path
    try:
        artifact = wandb.use_artifact(artifact_path, type="model")
        return artifact.download()
    except wandb.errors.CommError:
        st.error(f"Model artifact {artifact_path} not found ❌")
        return None

local_en_to_ar = "./models/en_to_ar"
local_ar_to_en = "./models/ar_to_en"

wandb.init(project="translation-finetuning", entity="abdelaziz67-ain-shams-university")
artifact_dir_en_to_ar = load_model(artifact_paths["en_to_ar"], local_en_to_ar)
artifact_dir_ar_to_en = load_model(artifact_paths["ar_to_en"], local_ar_to_en)

if artifact_dir_en_to_ar and artifact_dir_ar_to_en:
    model_en_to_ar = AutoModelForSeq2SeqLM.from_pretrained(artifact_dir_en_to_ar)
    tokenizer_en_to_ar = AutoTokenizer.from_pretrained(artifact_dir_en_to_ar)

    model_ar_to_en = AutoModelForSeq2SeqLM.from_pretrained(artifact_dir_ar_to_en)
    tokenizer_ar_to_en = AutoTokenizer.from_pretrained(artifact_dir_ar_to_en)

def translate(text, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(**inputs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

def recognize_and_translate():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        recognizer.energy_threshold = 300
        recognizer.pause_threshold = 0.5
        st.info(f"🎤 Listening for {speech_time} seconds...")
        
        try:
            audio = recognizer.listen(source, timeout=speech_time)
            text = recognizer.recognize_google(audio, language=input_lang)
            
            if not text or len(text.strip()) < 3:
                st.warning("Please speak more clearly")
                return
            
            st.success(f"Recognized: {text}")
            
            for lang in output_lang:
                if lang == "ar":
                    translated = translate(text, model_en_to_ar, tokenizer_en_to_ar)
                elif lang == "en":
                    translated = translate(text, model_ar_to_en, tokenizer_ar_to_en)
                else:
                    translated = translator.translate(text, dest=lang).text
                
                if translated:
                    st.write(f"{lang.upper()}: {translated}")
                    if translation_type == "Text + Speech":
                        threading.Thread(target=speak, args=(translated,)).start()
                else:
                    st.warning(f"Couldn't translate to {lang.upper()}")

        except sr.UnknownValueError:
            st.error("Couldn't understand audio")
        except sr.RequestError:
            st.error("Connection error")
        except Exception as e:
            st.error("An error occurred")

if st.button("Start Translation", type="primary"):
    recognize_and_translate()

wandb.finish()