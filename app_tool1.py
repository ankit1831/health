import streamlit as st
import numpy as np
import json
import os
import joblib
from groq import Groq
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
load_dotenv()
st.set_page_config(page_title="Heal Bridge AI Triage", page_icon="🩺", layout="centered")
client = Groq()

# --- 2. HIGH-SPEED ASSET LOADING ---
# --- 2. HIGH-SPEED ASSET LOADING ---
@st.cache_resource
def load_assets():
    model = joblib.load('heal_bridge_bnb_model.joblib')
    encoder = joblib.load('disease_label_encoder.joblib')
    
    with open('master_feature_columns.json', 'r', encoding='utf-8') as f:
        master_cols = json.load(f)
    with open('release_evidences.json', 'r', encoding='utf-8') as f:
        evidences = json.load(f)
    with open('symptom_hierarchy.json', 'r', encoding='utf-8') as f:
        hierarchy = json.load(f)
    # NEW: Load the Anatomical Categories
    with open('symptom_categories.json', 'r', encoding='utf-8') as f:
        categories = json.load(f)
        
    scribe_dict = {}
    for col in master_cols:
        if col in ['AGE', 'SEX'] or col.startswith('AGE_'): continue
        if '_@_' in col:
            base_code, val_code = col.split('_@_')
            base_q = evidences.get(base_code, {}).get('question_en', '')
            val_mean = evidences.get(base_code, {}).get('value_meaning', {}).get(val_code, {}).get('en', '')
            scribe_dict[col] = f"{base_q} -> {val_mean}"
        else:
            scribe_dict[col] = evidences.get(col, {}).get('question_en', col)
            
    english_to_code = {v.lower(): k for k, v in scribe_dict.items()}
    # Return the categories as well!
    return model, encoder, master_cols, evidences, hierarchy, categories, english_to_code

bnb_model, label_encoder, master_columns, evidences, hierarchy, categories, english_to_code = load_assets()

# --- 3. THE PATIENT TRACKER CLASS ---
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
                
    def get_known_positives(self):
        return [self.columns[idx] for idx, val in enumerate(self.state[0]) if val == 1 and self.columns[idx] not in ['SEX'] and not self.columns[idx].startswith('AGE')]

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

# --- 4. SESSION STATE (STATE MACHINE) ---
if "phase" not in st.session_state:
    st.session_state.phase = "onboarding" 
    st.session_state.messages = [{"role": "assistant", "content": "Hello. I am the Heal Bridge Triage Assistant. Before we begin, could you please tell me your age and biological gender?"}]
if "tracker" not in st.session_state: st.session_state.tracker = None
if "last_asked_text" not in st.session_state: st.session_state.last_asked_text = "None"
if "last_asked_code" not in st.session_state: st.session_state.last_asked_code = "None"
if "questions_asked" not in st.session_state: st.session_state.questions_asked = 0
# NEW: Temporary memory for onboarding
if "temp_age" not in st.session_state: st.session_state.temp_age = None
if "temp_sex" not in st.session_state: st.session_state.temp_sex = None

st.title("🩺 Heal Bridge Intake")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. THE MAIN CHAT ENGINE ---
if st.session_state.phase != "complete":
    if user_input := st.chat_input("Type your response here..."):
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)

        with st.spinner("Processing..."):
            
            # ==========================================
            # PHASE 1: ONBOARDING (AGE & GENDER)
            # ==========================================
            if st.session_state.phase == "onboarding":
                prompt = f"""Extract the user's age and biological sex from this text: "{user_input}". 
                Return a JSON object with keys 'age' (integer) and 'sex' ("M" or "F"). 
                If you cannot confidently determine one, set its value to null. Example: {{"age": 30, "sex": null}}"""
                
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.0, response_format={"type": "json_object"}
                )
                data = json.loads(completion.choices[0].message.content)
                
                # Save what it found to temporary memory
                if data.get('age') is not None: st.session_state.temp_age = data['age']
                if data.get('sex') is not None: st.session_state.temp_sex = data['sex']
                
                if st.session_state.temp_age is not None and st.session_state.temp_sex is not None:
                    # Success! Move to triage
                    is_male = True if st.session_state.temp_sex == 'M' else False
                    st.session_state.tracker = PatientTracker(age=st.session_state.temp_age, sex_is_male=is_male)
                    st.session_state.phase = "triage"
                    reply = f"Thank you. I have your profile set as a {st.session_state.temp_age}-year-old {'Male' if is_male else 'Female'}. Now, what brings you in today? Please describe your symptoms."
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                else:
                    # Dynamically ask for what is missing
                    missing = []
                    if st.session_state.temp_age is None: missing.append("age")
                    if st.session_state.temp_sex is None: missing.append("biological gender (M/F)")
                    reply = f"Got it. To finish setting up your profile, I just need your **{' and '.join(missing)}**."
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()

            ## ==========================================
            # PHASE 2: MEDICAL TRIAGE
            # ==========================================
            elif st.session_state.phase == "triage":
                
                # 1. Smarter "Direct-to-Code" Scribe
                # We feed the LLM a mini-dictionary of just the 223 parent codes so it doesn't have to guess English strings!
                core_dict = {c: evidences[c].get("question_en", c) for c in hierarchy.keys() if c in master_columns}
                
                scribe_prompt = f"""You are an expert Medical Extraction AI. 
                NURSE ASKED: '{st.session_state.last_asked_text}' (Code: {st.session_state.last_asked_code})
                PATIENT REPLIED: '{user_input}'
                
                TASK 1: Did the patient answer the Nurse's question? If so, output the Nurse's Code with 1 (Yes) or 0 (No).
                TASK 2: Did the patient mention NEW symptoms? Find the best matching codes from this dictionary:
                {json.dumps(core_dict)}
                
                CRITICAL RULES: 
                - If the patient mentions ANY specific pain (headache, chest pain, stomach ache), you MUST output "E_55": 1.
                - Ignore the Nurse's Code if it is 'None'.
                
                Output ONLY a JSON object mapping codes to 1 or 0. Example: {{"E_55": 1, "E_91": 0}}"""
                
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "system", "content": scribe_prompt}],
                    temperature=0.0, response_format={"type": "json_object"}
                )
                
                raw_json = json.loads(completion.choices[0].message.content)
                # Safely filter out hallucinations
                valid_json = {k: v for k, v in raw_json.items() if k in master_columns}
                st.session_state.tracker.update_symptoms(valid_json)

                # 2. Prediction & Decision Logic
                probabilities = calculate_true_confidence(st.session_state.tracker.state, bnb_model)
                top_id = np.argsort(probabilities)[::-1][0]
                top_confidence = probabilities[top_id]

                if top_confidence >= 0.50 and len(st.session_state.tracker.get_known_positives()) > 1:
                    st.session_state.messages.append({"role": "assistant", "content": f"✅ **Triage Complete!**\nThe engine is {top_confidence * 100:.1f}% confident the condition is: **{label_encoder.classes_[top_id]}**"})
                    st.session_state.phase = "complete"
                    st.rerun()
                    
                if st.session_state.questions_asked >= 8: # Bumped to 8 to give the AI breathing room
                    st.session_state.messages.append({"role": "assistant", "content": f"⚠️ **Safety Net Triggered.**\nHighest confidence is {top_confidence * 100:.1f}%. Symptoms are too complex. Recommending physical physician evaluation."})
                    st.session_state.phase = "complete"
                    st.rerun()

                # 3. Entropy Engine
                current_probs_clipped = np.clip(probabilities, 1e-10, 1.0)
                current_entropy = -np.sum(current_probs_clipped * np.log2(current_probs_clipped))
                
                legal_questions = []
                known_positives = st.session_state.tracker.get_known_positives()
                
                for col in master_columns:
                    if col == 'AGE' or col == 'SEX' or col.startswith('AGE_'): continue
                    if st.session_state.tracker.state[0, master_columns.index(col)] != -1: continue
                    if col in hierarchy: legal_questions.append(col)
                    elif '_@_' in col and col.split('_@_')[0] in known_positives: legal_questions.append(col)

                def calc_h(idx):
                    p_yes = np.exp(bnb_model.feature_log_prob_[:, idx])
                    p_no = 1.0 - p_yes
                    p_user_yes = np.sum(probabilities * p_yes)
                    
                    s_yes = st.session_state.tracker.state.copy()
                    s_yes[0, idx] = 1
                    prob_y = np.clip(calculate_true_confidence(s_yes, bnb_model), 1e-10, 1.0)
                    h_yes = -np.sum(prob_y * np.log2(prob_y))
                    
                    s_no = st.session_state.tracker.state.copy()
                    s_no[0, idx] = 0
                    prob_n = np.clip(calculate_true_confidence(s_no, bnb_model), 1e-10, 1.0)
                    h_no = -np.sum(prob_n * np.log2(prob_n))
                    
                    return (p_user_yes * h_yes) + ((1.0 - p_user_yes) * h_no)

                # --- NEW: TRAIN OF THOUGHT LOGIC ---
                # Figure out which organ systems are currently "active" based on known positives
                active_categories = set()
                for pos_code in known_positives:
                    base_pos = pos_code.split('_@_')[0]
                    if base_pos in categories:
                        active_categories.add(categories[base_pos])

                info_gains = []
                for q_code in legal_questions:
                    idx = master_columns.index(q_code)
                    raw_gain = current_entropy - calc_h(idx)
                    
                    base_q = q_code.split('_@_')[0]
                    q_category = categories.get(base_q)
                    
                    mult = 1.0
                    if '_@_' in q_code:
                        mult = 5.0 # Priority 1: Specific follow-ups to active symptoms
                    elif q_category in active_categories:
                        mult = 3.0 # Priority 2: TRAIN OF THOUGHT! Stay in the same organ system.
                    elif q_code in ['E_55', 'E_201', 'E_147', 'E_51', 'E_129', 'E_228', 'E_133']:
                        mult = 1.1 # Priority 3: Standard core triage questions
                    elif q_code in ['E_149', 'E_226', 'E_93']:
                        mult = 0.2 # Penalty: Edge cases
                        
                    info_gains.append((q_code, raw_gain * mult))

                info_gains.sort(key=lambda x: x[1], reverse=True)
                best_code = info_gains[0][0]
                
                if '_@_' in best_code:
                    base_code, val_code = best_code.split('_@_')
                    best_human = f"{evidences.get(base_code, {}).get('question_en', '')} -> {evidences.get(base_code, {}).get('value_meaning', {}).get(val_code, {}).get('en', '')}"
                else:
                    best_human = evidences.get(best_code, {}).get('question_en', best_code)

                st.session_state.tracker.state[0, master_columns.index(best_code)] = 0 
                st.session_state.last_asked_text = best_human
                st.session_state.last_asked_code = best_code

                nurse_reply = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "system", "content": f"You are a warm, empathetic triage nurse. Ask the patient if they are experiencing this specific symptom: '{best_human}'. Keep it brief and conversational."}],
                    temperature=0.5
                ).choices[0].message.content
                
                st.session_state.messages.append({"role": "assistant", "content": nurse_reply})
                st.session_state.questions_asked += 1
                st.rerun()

else:
    if st.button("Restart Intake"):
        st.session_state.clear()
        st.rerun()