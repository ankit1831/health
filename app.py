import streamlit as st
from groq import Groq
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json
import re
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()
st.set_page_config(page_title="Heal Bridge AI Triage", page_icon="🩺", layout="centered")

DISEASES = [
    'Acute COPD exacerbation / infection', 'Acute dystonic reactions', 'Acute laryngitis', 
    'Acute otitis media', 'Acute pulmonary edema', 'Acute rhinosinusitis', 'Allergic sinusitis', 
    'Anaphylaxis', 'Anemia', 'Atrial fibrillation', 'Boerhaave', 'Bronchiectasis', 
    'Bronchiolitis', 'Bronchitis', 'Bronchospasm / acute asthma exacerbation', 'Chagas', 
    'Chronic rhinosinusitis', 'Cluster headache', 'Croup', 'Ebola', 'Epiglottitis', 'GERD', 
    'Guillain-Barré syndrome', 'HIV (initial infection)', 'Influenza', 'Inguinal hernia', 
    'Larygospasm', 'Localized edema', 'Myasthenia gravis', 'Myocarditis', 'PSVT', 
    'Pancreatic neoplasm', 'Panic attack', 'Pericarditis', 'Pneumonia', 'Possible NSTEMI / STEMI', 
    'Pulmonary embolism', 'Pulmonary neoplasm', 'SLE', 'Sarcoidosis', 'Scombroid food poisoning', 
    'Spontaneous pneumothorax', 'Spontaneous rib fracture', 'Stable angina', 'Tuberculosis', 
    'URTI', 'Unstable angina', 'Viral pharyngitis', 'Whooping cough'
]

# --- LOAD EXACT DATASET VOCABULARY ---
@st.cache_data
def load_ddxplus_vocabulary():
    try:
        with open('release_evidences.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        symptoms = []
        antecedents = []
        for key, value in data.items():
            question = value.get("question_en", "")
            if not question: continue
            if value.get("is_antecedent"):
                antecedents.append(question)
            else:
                symptoms.append(question)
        return symptoms[:80], antecedents[:20] 
    except FileNotFoundError:
        return ["pain", "cough", "fever", "shortness of breath", "rash"], ["Do you smoke?"]

symptoms_list, antecedents_list = load_ddxplus_vocabulary()

# --- THE MASTER PROMPT (TUNED FOR LLAMA 3) ---
# --- EXPERIMENT A.2: THE FEW-SHOT MASTER PROMPT ---
SYSTEM_PROMPT = f"""
You are an elite Clinical Triage Diagnostician conducting a strict, multi-turn interview.

CRITICAL PROTOCOL:
1. ONE QUESTION AT A TIME: Ask exactly ONE follow-up question per response to gather positive and negative evidence.
2. DYNAMIC QUESTIONING: Based on the chief complaint, formulate a hypothesis. Ask targeted questions from this list: {', '.join(symptoms_list[:80])} and this history list: {', '.join(antecedents_list[:20])}.
3. MANDATORY LENGTH: Ask 5 to 10 questions.
4. THE BAILOUT TRIGGER: If the user explicitly says "enough", "stop", or "predict now", you MUST immediately stop the interview and output the JSON block.

CRITICAL JSON EXTRACTION RULES (DO NOT FAIL THESE):
When you output the JSON, you MUST NOT write conversational summaries. You must COPY AND PASTE the EXACT STRING from the allowed lists. 

MAPPING EXAMPLES:
- If user says "sometimes" to choking -> You output EXACTLY: "Do you feel like you are (or were) choking or suffocating?"
- If user says "yeah" to deep breathing pain -> You output EXACTLY: "Do you have pain that is increased when you breathe in deeply?"
- If user says "not at all" to shortness of breath -> You put EXACTLY "Are you experiencing shortness of breath or difficulty breathing in a significant way?" into the negative_symptoms array.

JSON FORMAT TO OUTPUT (ABSOLUTELY NO OTHER TEXT):
{{
  "age": 30,
  "sex": "male",
  "chief_complaint": "Exact string from symptom list",
  "positive_symptoms": ["Exact string 1", "Exact string 2"],
  "negative_symptoms": ["Exact string 3", "Exact string 4"],
  "medical_history": [
    {{"question": "Exact string from antecedent list", "answer": "yes"}}
  ]
}}
"""

@st.cache_resource 
def load_ai_engine():
    device = torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModelForSequenceClassification.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT",
        num_labels=49, problem_type="multi_label_classification"
    )
    checkpoint = torch.load("saved_models/heal_bridge_best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return tokenizer, model, device

# --- INITIALIZE GROQ SESSION ---
if "groq_client" not in st.session_state:
    st.session_state.groq_client = Groq() # Automatically looks for GROQ_API_KEY in environment

if "messages" not in st.session_state:
    # Groq requires the system prompt to be the first message in the array
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Hello. I am the Heal Bridge Triage Assistant. What brings you in today?"}
    ]
    st.session_state.triage_complete = False 

st.title("🩺 Heal Bridge Intake")

# Draw the chat (skip the invisible system prompt)
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if not st.session_state.triage_complete:
    if user_input := st.chat_input("Type your symptoms here..."):
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing and formulating next question..."):
                try:
                    # Send the entire message array to Groq
                    chat_completion = st.session_state.groq_client.chat.completions.create(
                        messages=st.session_state.messages,
                        model="llama-3.3-70b-versatile",
                        temperature=0.0
                    )
                    ai_reply = chat_completion.choices[0].message.content.strip()
                    
                    # Intercept JSON output
                    if "{" in ai_reply and '"age"' in ai_reply.lower():
                        st.session_state.triage_complete = True
                        json_str = re.search(r'\{.*\}', ai_reply.replace('\n', ' '), re.DOTALL)
                        if json_str:
                            try:
                                data = json.loads(json_str.group(0))
                                
                                # --- NLP TEXT CLEANER ---
                                def clean_text(text):
                                    text = re.sub(r'^(Do you have|Are you experiencing|Do you feel like you are \(or were\)|Do you feel like you are|Have you had) ', '', text, flags=re.IGNORECASE)
                                    text = text.replace('?', '').replace('your ', 'my ').replace('any ', '')
                                    return text.strip().lower()

                                # --- THE PYTHON SYNTHESIZER (NOW WITH NEGATIVE EVIDENCE) ---
                                cc_clean = clean_text(data.get('chief_complaint', 'pain'))
                                synthetic_text = f"I am an {data.get('age', 30)}-year-old {data.get('sex', 'male')}. "
                                synthetic_text += f"I came into the clinic today because I have {cc_clean}. "
                                
                                for symp in data.get('positive_symptoms', []):
                                    synthetic_text += f"To give you more context: I have {clean_text(symp)}. "
                                    
                                for neg in data.get('negative_symptoms', []):
                                    synthetic_text += f"I do not have {clean_text(neg)}. "
                                    
                                for hist in data.get('medical_history', []):
                                    synthetic_text += f"Regarding the question '{hist.get('question')}', my answer is {hist.get('answer')}. "
                                    
                                with st.spinner("Loading Deep Learning Engine & Analyzing..."):
                                    tokenizer, torch_model, device = load_ai_engine()
                                    encoding = tokenizer(
                                        synthetic_text, add_special_tokens=True, max_length=512,
                                        padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
                                    )
                                    with torch.no_grad():
                                        outputs = torch_model(input_ids=encoding['input_ids'].to(device), attention_mask=encoding['attention_mask'].to(device))
                                        probabilities = torch.sigmoid(outputs.logits)[0]
                                        
                                    top_k_probs, top_k_indices = torch.topk(probabilities, 3)
                                    
                                    report = "### 🩺 Diagnostic Predictions:\n"
                                    report += f"**Synthesized Data Passed to PyTorch:**\n\n> *{synthetic_text}*\n\n"
                                    for i in range(3):
                                        report += f"**{i+1}. {DISEASES[top_k_indices[i].item()]}** ({top_k_probs[i].item()*100:.2f}%)\n"
                                    
                                st.session_state.messages.append({"role": "assistant", "content": report})
                                st.rerun()
                            except json.JSONDecodeError:
                                st.error("AI JSON parsing failed.")
                                st.write("Raw Output:", ai_reply)
                    else:
                        st.markdown(ai_reply)
                        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                        
                except Exception as e:
                    st.error(f"An unexpected API error occurred: {e}")