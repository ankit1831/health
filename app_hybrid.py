import streamlit as st
import numpy as np
import json
import joblib
import os
import re
import time
from groq import Groq
from dotenv import load_dotenv
import google.generativeai as genai

# --- GEMINI CONFIGURATION & CMO PROMPTS ---
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    genai.configure(api_key=gemini_key)

TRIAGE_SEVERITY = {
    'Anaphylaxis': 'RED', 'Acute pulmonary edema': 'RED', 'Boerhaave': 'RED', 
    'Ebola': 'RED', 'Epiglottitis': 'RED', 'Possible NSTEMI / STEMI': 'RED', 
    'Pulmonary embolism': 'RED', 'Spontaneous pneumothorax': 'RED', 
    'Unstable angina': 'RED', 'Larygospasm': 'RED', 'Myocarditis': 'RED', 
    'Guillain-Barré syndrome': 'RED', 'PSVT': 'RED', 
    'Bronchospasm / acute asthma exacerbation': 'RED',
    'Acute COPD exacerbation / infection': 'YELLOW', 'Acute dystonic reactions': 'YELLOW', 
    'Atrial fibrillation': 'YELLOW', 'Bronchiectasis': 'YELLOW', 'Bronchiolitis': 'YELLOW', 
    'Chagas': 'YELLOW', 'Croup': 'YELLOW', 'HIV (initial infection)': 'YELLOW', 
    'Inguinal hernia': 'YELLOW', 'Myasthenia gravis': 'YELLOW', 'Pancreatic neoplasm': 'YELLOW', 
    'Pericarditis': 'YELLOW', 'Pneumonia': 'YELLOW', 'Pulmonary neoplasm': 'YELLOW', 
    'SLE': 'YELLOW', 'Sarcoidosis': 'YELLOW', 'Spontaneous rib fracture': 'YELLOW', 
    'Stable angina': 'YELLOW', 'Tuberculosis': 'YELLOW', 'Whooping cough': 'YELLOW', 
    'Scombroid food poisoning': 'YELLOW', 'Anemia': 'YELLOW', 'Cluster headache': 'YELLOW',
    'Acute laryngitis': 'GREEN', 'Acute otitis media': 'GREEN', 'Acute rhinosinusitis': 'GREEN', 
    'Allergic sinusitis': 'GREEN', 'Bronchitis': 'GREEN', 'Chronic rhinosinusitis': 'GREEN', 
    'GERD': 'GREEN', 'Influenza': 'GREEN', 'Localized edema': 'GREEN', 
    'Panic attack': 'GREEN', 'URTI': 'GREEN', 'Viral pharyngitis': 'GREEN'
}

def evaluate_clinical_severity(top_3_diseases):
    primary_disease = top_3_diseases[0]
    severity = TRIAGE_SEVERITY.get(primary_disease, "YELLOW")
    if severity == "RED": return "RED", "[SYSTEM ALERT: CRITICAL EMERGENCY. Instruct the patient to go to the nearest ER. DO NOT suggest home remedies.]"
    elif severity == "YELLOW": return "YELLOW", "[SYSTEM ALERT: URGENT CONDITION. Instruct the patient to schedule an appointment with a doctor soon. Provide safe temporary management.]"
    else: return "GREEN", "[SYSTEM ALERT: MINOR CONDITION. Reassure the patient. Provide safe over-the-counter home care.]"

def generate_cmo_prompt(top_3_diseases, system_alert, patient_text):
    return f"""You are the Chief Medical Officer, a highly empathetic clinical AI assistant.
PATIENT'S COMPLAINT HISTORY: {patient_text}

THE DIAGNOSTIC ENGINE'S PREDICTION: {', '.join(top_3_diseases)}. 
🚨 CRITICAL SAFETY OVERRIDE: {system_alert}

YOUR RULES:
1. Speak directly to the patient with a warm bedside manner.
2. CRITICAL: You MUST explicitly state the top condition predicted by the Diagnostic Engine in your first sentence (e.g., "Based on the triage engine's analysis, it looks like you might be dealing with..."). Do not invent your own diagnosis.
3. Explain that top condition in simple, non-medical terms.
4. DO NOT recommend specific drug dosages.
5. End your message by asking: "Do you have any questions about this condition or what to do next?"
"""

# --- 1. SETUP & CONFIGURATION ---
load_dotenv()
st.set_page_config(page_title="Heal Bridge AI Triage", page_icon="🩺", layout="centered")

# We will use Groq for the blazing fast chat, just like your Cerebras setup
client = Groq()

# --- 2. ASSET LOADING (Replacing PyTorch with Naive Bayes Assets) ---
@st.cache_resource
def load_assets():
    model = joblib.load('heal_bridge_bnb_model.joblib')
    encoder = joblib.load('disease_label_encoder.joblib')
    
    with open('master_feature_columns.json', 'r', encoding='utf-8') as f:
        master_cols = json.load(f)
    with open('release_evidences.json', 'r', encoding='utf-8') as f:
        evidences = json.load(f)
        
    # Build the vocabulary list for the LLM Prompt (exactly like your load_ddxplus_vocabulary function)
    vocab_list = []
    for key, value in evidences.items():
        question = value.get("question_en", "")
        if question: vocab_list.append(question)
            
    return model, encoder, master_cols, evidences, vocab_list

bnb_model, label_encoder, master_columns, evidences, symptoms_list = load_assets()
# --- THE MATH ENGINE COMPONENTS ---
class PatientTracker:
    def __init__(self, age, sex_is_male):
        self.columns = master_columns
        self.state = np.full((1, len(self.columns)), -1, dtype=np.int8)
        
        age_bins = ['AGE_0_18', 'AGE_19_35', 'AGE_36_50', 'AGE_51_65', 'AGE_66_PLUS']
        for bin_name in age_bins: self.state[0, self.columns.index(bin_name)] = 0 
            
        if age <= 18: self.state[0, self.columns.index('AGE_0_18')] = 1
        elif age <= 35: self.state[0, self.columns.index('AGE_19_35')] = 1
        elif age <= 50: self.state[0, self.columns.index('AGE_36_50')] = 1
        elif age <= 65: self.state[0, self.columns.index('AGE_51_65')] = 1
        else: self.state[0, self.columns.index('AGE_66_PLUS')] = 1
            
        self.state[0, self.columns.index('SEX')] = 1 if sex_is_male else 0
        
    def update_symptoms(self, extracted_json):
        for code, value in extracted_json.items():
            if code in self.columns:
                self.state[0, self.columns.index(code)] = value

def calculate_true_confidence(tracker_state, model):
    log_probs = model.class_log_prior_.copy()
    log_prob_1 = model.feature_log_prob_
    prob_1 = np.exp(log_prob_1)
    log_prob_0 = np.log(np.clip(1.0 - prob_1, 1e-10, 1.0)) 
    
    for idx, val in enumerate(tracker_state[0]):
        if val == 1: log_probs += log_prob_1[:, idx]
        elif val == 0: log_probs += log_prob_0[:, idx]
            
    log_probs_shifted = log_probs - np.max(log_probs)
    probs = np.exp(log_probs_shifted)
    return probs / np.sum(probs)
# --- 3. SESSION STATE INITIALIZATION (Exactly from your code) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello. I am the Heal Bridge Triage Assistant. Could you please tell me your age, biological sex, and what brings you in today?"}
    ]
    st.session_state.triage_complete = False 

if "patient_record" not in st.session_state:
    st.session_state.patient_record = {"age": None, "sex": None}
    
if "demo_ask_count" not in st.session_state: st.session_state.demo_ask_count = 0
if "next_intercept_turn" not in st.session_state: st.session_state.next_intercept_turn = 6
if "force_predict" not in st.session_state: st.session_state.force_predict = False

st.title("🩺 Heal Bridge Intake")

# --- 4. THE UI ESCAPE HATCH (Your Sidebar) ---
with st.sidebar:
    st.markdown("### Patient Controls")
    if st.button("🔄 Start New Assessment", type="primary", use_container_width=True):
        st.session_state.clear()
        st.rerun()
        
    if not st.session_state.get("triage_complete"):
        if st.button("🩺 Generate Diagnosis Now", use_container_width=True):
            st.session_state.force_predict = True
            st.rerun()

# --- 5. RENDER CHAT HISTORY ---
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- 6. THE CHAT ENGINE & ROUTING ---
if not st.session_state.triage_complete:
    
    # Calculate if they have hit the rolling limit (Your Intercept Logic)
    user_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
    limit_reached = (user_turns >= st.session_state.next_intercept_turn)
    
    if limit_reached:
        st.info("💡 I have gathered a detailed clinical picture. Would you like to proceed to the diagnostic results, or do you have more symptoms to share?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🩺 Generate Diagnosis", use_container_width=True):
                st.session_state.force_predict = True
                st.session_state.next_intercept_turn += 100 
                st.rerun()
        with col2:
            if st.button("💬 Add More Symptoms", use_container_width=True):
                st.session_state.next_intercept_turn += 5
                st.rerun()
                
    else:
        user_input = st.chat_input("Type your symptoms here...")
        
        if user_input or st.session_state.force_predict:
            
            if st.session_state.force_predict and not user_input:
                user_input = "predict now"
                st.session_state.messages.append({"role": "user", "content": "I am ready for my diagnosis now."})
                with st.chat_message("user"): st.markdown("*User requested immediate diagnosis.*")
            else:
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"): st.markdown(user_input)
                    
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    
                    # --- STATE ROUTER LOGIC ---
                    is_missing_demographics = (st.session_state.patient_record["age"] is None)
                    user_typed_stop = any(trigger in user_input.lower() for trigger in ["predict now", "stop", "enough"])
                    total_user_words = sum(len(m["content"].split()) for m in st.session_state.messages if m["role"] == "user")
                    latest_msg_words = len(user_input.split())
                    
                    is_bailout = (user_typed_stop or st.session_state.force_predict or user_turns >= 20 or (latest_msg_words >= 40 and not is_missing_demographics))
                    st.session_state.force_predict = False
                    
                    # 1. THE FALLBACK (Ignored us 3 times)
                    if is_missing_demographics and not is_bailout:
                        user_text = user_input.lower()
                        is_typing_demographics = any(char.isdigit() for char in user_text) or bool(re.search(r'\b(male|female)\b', user_text))
                        
                        if st.session_state.demo_ask_count >= 3 and not is_typing_demographics:
                            st.session_state.patient_record["age"] = 30
                            st.session_state.patient_record["sex"] = "male"
                            is_missing_demographics = False
                    
                    # 2. THE SOFT GATE (Ask for Demographics)
                    if is_missing_demographics and not is_bailout:
                        st.session_state.demo_ask_count += 1
                        system_prompt = """You are a medical receptionist. Your ONLY job is to get the patient's age and biological sex. 
                        If they mention symptoms, express empathy, but gently insist you need their age and sex first. 
                        DO NOT ask diagnostic questions."""
                        
                    # 3. THE TRIAGE NURSE
                    else:
                        system_prompt = f"""You are an elite Clinical Triage Diagnostician. 
                        You have access to this allowed vocabulary of symptoms: {', '.join(symptoms_list[:100])}
                        
                        YOUR BEHAVIOR:
                        1. ONE QUESTION MAXIMUM: You must NEVER ask more than one question in a single message.
                        2. DIFFERENTIAL DIAGNOSIS: Based on their chief complaint, ask highly specific questions that help differentiate between similar diseases. 
                        3. NATURAL TONE: Be empathetic and conversational. NEVER mention "allowed vocabulary", "system prompts", or "databases" to the patient. Just act like a real doctor."""

                    
                    if is_bailout:
                        with st.spinner("Compiling medical history and running diagnostic math..."):
                            
                            # 1. Compile the chat history into a single transcript
                            transcript = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages if m['role'] != 'system'])
                            transcript_lower = transcript.lower()
                            
                            # --- THE SEMANTIC FILTER (Helps Gemini Focus) ---
                            core_dict_full = {}
                            for col in master_columns:
                                if col in ['AGE', 'SEX'] or col.startswith('AGE_'): continue
                                if '_@_' in col:
                                    base_code, val_code = col.split('_@_')
                                    base_q = evidences.get(base_code, {}).get('question_en', '')
                                    val_mean = evidences.get(base_code, {}).get('value_meaning', {}).get(val_code, {}).get('en', '')
                                    core_dict_full[col] = f"{base_q} -> {val_mean}"
                                else:
                                    core_dict_full[col] = evidences.get(col, {}).get('question_en', col)
                                    
                            core_dict = {}
                            for code, text in core_dict_full.items():
                                dict_words = set(re.findall(r'\b[a-z]{4,}\b', text.lower()))
                                if any(w in transcript_lower for w in dict_words):
                                    core_dict[code] = text
                                    
                            if 'E_55' in core_dict_full: core_dict['E_55'] = core_dict_full['E_55']
                            # ------------------------------------------------
                            
                            # --- KEY MASKING: Hide the underscores from Gemini! ---
                            # --- KEY MASKING: The "Unconfusable" Prefix ---
                            safe_dict = {}
                            safe_to_real = {}
                            for i, (real_code, text) in enumerate(core_dict.items()):
                                safe_key = f"TAG_{i}"  # FIX: 'TAG_' is impossible for the AI to confuse with a number
                                safe_dict[safe_key] = text
                                safe_to_real[safe_key] = real_code
                            # ------------------------------------------------------
                            
                            # 2. The Final Extraction Prompt
                            extraction_prompt = f"""You are an expert Medical Scribe. 
                            Review this triage conversation:
                            
                            {transcript}
                            
                            Here is your allowed dictionary of symptoms:
                            {json.dumps(safe_dict, indent=2)}
                            
                            CRITICAL RULES:
                            1. THINK FIRST: Read carefully for negations. If they say "no nausea", it goes in the denied list. 
                            2. EXACT LOCATIONS: Pay close attention to exactly where the pain is and find the precise location code.
                            3. STRICT STRINGS: ONLY output the exact strings provided (e.g., "TAG_15"). Do not change the letters.
                            
                            Output ONLY a JSON object in this EXACT format:
                            {{
                                "reasoning": "Patient explicitly denied nausea (TAG_12). Confirmed stomach pain (TAG_4).",
                                "confirmed_codes": ["TAG_4", "TAG_8"],
                                "denied_codes": ["TAG_12"]
                            }}"""
                            
                            try:
                                # Send to Gemini 2.5 Flash
                                scribe_model = genai.GenerativeModel(
                                    model_name="gemini-2.5-flash",
                                    generation_config={"response_mime_type": "application/json", "temperature": 0.0}
                                )
                                extraction = scribe_model.generate_content(extraction_prompt)
                                
                                # Parse the Array format
                                full_json = json.loads(extraction.text)
                                confirmed_safe = full_json.get("confirmed_codes", [])
                                denied_safe = full_json.get("denied_codes", [])
                                
                                # --- UNMASK THE KEYS FOR THE MATH ENGINE ---
                                valid_json = {}
                                for scode in confirmed_safe:
                                    real_code = safe_to_real.get(scode)
                                    if real_code and real_code in master_columns: 
                                        valid_json[real_code] = 1
                                        
                                for scode in denied_safe:
                                    real_code = safe_to_real.get(scode)
                                    if real_code and real_code in master_columns: 
                                        valid_json[real_code] = 0
                                # -------------------------------------------
                                
                                # Save the FULL JSON so we can read the AI's mind on screen!
                                st.session_state.debug_extracted_json = full_json
                                
                                # 3. Initialize Tracker and Run Math
                                pat_age = st.session_state.patient_record["age"] if st.session_state.patient_record["age"] is not None else 30
                                pat_sex = st.session_state.patient_record["sex"] if st.session_state.patient_record["sex"] is not None else "male"
                                
                                is_male = True if pat_sex.lower() == "male" else False
                                tracker = PatientTracker(age=pat_age, sex_is_male=is_male)
                                tracker.update_symptoms(valid_json)
                                
                                probs = calculate_true_confidence(tracker.state, bnb_model)
                                top_3_indices = np.argsort(probs)[::-1][:3]
                                
                                # 4. Save the Results to Memory!
                                st.session_state.top_3_results = []
                                top_3_names = []
                                for i, idx in enumerate(top_3_indices):
                                    disease = label_encoder.classes_[idx]
                                    top_3_names.append(disease)
                                    confidence = probs[idx] * 100
                                    st.session_state.top_3_results.append(f"**{i+1}. {disease}** ({confidence:.1f}% confidence)")
                                
                                # 5. Fire up the CMO (Gemini)
                                severity, alert = evaluate_clinical_severity(top_3_names)
                                cmo_model = genai.GenerativeModel(
                                    model_name="gemini-2.5-flash", 
                                    system_instruction=generate_cmo_prompt(top_3_names, alert, transcript)
                                )
                                st.session_state.cmo_chat = cmo_model.start_chat(history=[])
                                initial_response = st.session_state.cmo_chat.send_message("Please give me my triage results and explain what they mean.")
                                st.session_state.cmo_messages = [{"role": "assistant", "content": initial_response.text}]
                                st.session_state.current_phase = "CMO_QA"
                                
                                st.session_state.triage_complete = True
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Diagnostic Engine Error: {e}")
                    
                    # 5. THE NORMAL CHAT FLOW (If we aren't diagnosing yet)
                    else:
                        try:
                            # Using Llama 3.1 8b for blazing fast conversational speed
                            chat_completion = client.chat.completions.create(
                                model="llama-3.1-8b-instant",
                                messages=[{"role": "system", "content": system_prompt}] + st.session_state.messages,
                                temperature=0.5
                            )
                            ai_reply = chat_completion.choices[0].message.content.strip()
                            
                            # Minimal regex to catch age/sex if the AI receptionist successfully got it
                            if is_missing_demographics and st.session_state.demo_ask_count > 0:
                                age_match = re.search(r'\b(\d{1,3})\b', user_input)
                                sex_match = re.search(r'\b(male|female)\b', user_input.lower())
                                if age_match and sex_match:
                                    st.session_state.patient_record["age"] = int(age_match.group(1))
                                    st.session_state.patient_record["sex"] = sex_match.group(1)
                                    ai_reply = "Thank you. I have recorded your details. Let's talk about your symptoms. Can you tell me more about what you are experiencing?"

                            st.markdown(ai_reply)
                            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                            
                        except Exception as e:
                            st.error(f"API Error: {e}")


# --- 7. FINAL DIAGNOSIS SCREEN & CMO Q&A ---
if st.session_state.triage_complete:
    st.success("✅ Diagnostic Engine Complete. Transferring to the Chief Medical Officer...")
    
    # --- FIX: Draw the debug box here so it stays permanently ---
    if "debug_extracted_json" in st.session_state:
        st.info("🔍 **System Debug - Symptoms Extracted by Gemini:**")
        st.json(st.session_state.debug_extracted_json)
    # ------------------------------------------------------------
    
    st.markdown("### Top Suspected Conditions:")
    for result in st.session_state.top_3_results:
        st.write(result)
    st.markdown("---")

    if st.session_state.get("current_phase") == "CMO_QA":
        # 1. Draw the CMO Conversation History
        for msg in st.session_state.cmo_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # 2. DYNAMIC ACTION CHIPS
        if st.session_state.cmo_messages[-1]["role"] == "assistant":
            st.write("💡 **Suggested Questions:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("💊 What are my treatment options?", use_container_width=True):
                    st.session_state.chip_trigger = "What are my treatment options?"
                    st.rerun()
            with col2:
                if st.button("🍲 What foods should I eat or avoid?", use_container_width=True):
                    st.session_state.chip_trigger = "What foods should I eat or avoid?"
                    st.rerun()
            with col3:
                # FIX: Added a unique 'key' to prevent the duplicate ID crash
                if st.button("🔄 Start New Assessment", type="primary", use_container_width=True, key="cmo_restart"):
                    st.session_state.clear()
                    st.rerun()

        # 3. The Follow-Up Chat Box
        cmo_input = st.chat_input("Ask a follow-up question about your results...")
        
        if st.session_state.get("chip_trigger"):
            cmo_input = st.session_state.chip_trigger
            del st.session_state.chip_trigger  
            
        if cmo_input:
            st.session_state.cmo_messages.append({"role": "user", "content": cmo_input})
            with st.chat_message("user"): st.markdown(cmo_input)
                
            with st.chat_message("assistant"):
                with st.spinner("The Chief Medical Officer is typing..."):
                    try:
                        cmo_response = st.session_state.cmo_chat.send_message(cmo_input)
                        st.markdown(cmo_response.text)
                        st.session_state.cmo_messages.append({"role": "assistant", "content": cmo_response.text})
                        st.rerun() 
                    except Exception as e:
                        st.error(f"CMO Connection Error: {e}")