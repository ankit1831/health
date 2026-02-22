import streamlit as st
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# 1. Load API Key
load_dotenv()

st.set_page_config(page_title="Heal Bridge AI Triage", page_icon="🩺", layout="centered")

# 2. The Strict Dataset Vocabulary
DDXPLUS_SYMPTOMS = [
    "abdominal pain", "chest pain", "cough", "fever", "shortness of breath",
    "dizziness", "headache", "nausea", "vomiting", "sore throat",
    "palpitations", "rash", "fatigue", "severe allergic reaction", "edema of the throat"
]

# 3. The "State Machine" System Prompt
SYSTEM_PROMPT = f"""
You are an expert clinical triage nurse. Your goal is to collect the patient's Age, Sex, and Symptoms.
CRITICAL RULES:
1. Be empathetic but very brief. Ask follow-up questions if Age, Sex, or Symptoms are missing.
2. Map the patient's natural symptoms ONLY to the exact terms in this list: {', '.join(DDXPLUS_SYMPTOMS)}. Do not use any other medical vocabulary.
3. Once you have collected the Age, Sex, and at least one mapped symptom, you must immediately stop the conversation.
4. When you have all the information, your final output MUST be exactly this string, with absolutely no other text, greetings, or filler:
[TRIGGER_DIAGNOSIS: I am a [Age]-year-old [Sex]. I came into the clinic complaining of [Mapped Symptoms].]
"""

# --- THE FIX: Lock the Client Connection into Memory ---
if "gemini_client" not in st.session_state:
    st.session_state.gemini_client = genai.Client()

# 4. Initialize the Gemini Chat Session using the saved Client
if "chat_session" not in st.session_state or "messages" not in st.session_state:
    
    st.session_state.chat_session = st.session_state.gemini_client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1 # Low temperature keeps it strictly following the rules
        )
    )
    
    greeting = "Hello. I am the Heal Bridge Triage Assistant. Could you please tell me what brings you in today?"
    st.session_state.messages = [{"role": "assistant", "content": greeting}]

# --- USER INTERFACE ---
st.title("🩺 Heal Bridge Intake")
st.markdown("Chat with our AI nurse to begin your diagnosis.")

# 5. Display the Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. The Chat Input Box
if user_input := st.chat_input("Type your symptoms here..."):
    
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        with st.spinner("Typing..."):
            
            # Send message using the saved chat session
            response = st.session_state.chat_session.send_message(user_input)
            ai_reply = response.text.strip()
            
            # --- THE INTERCEPTOR LOGIC ---
            if "[TRIGGER_DIAGNOSIS" in ai_reply:
                st.success("✅ Triage Complete! Data extracted successfully.")
                st.info(f"**Hidden Data Passed to PyTorch:** {ai_reply}")
            else:
                st.markdown(ai_reply)
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})