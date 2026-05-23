import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import streamlit as st
import time
import numpy as np
import json
import joblib
import os
import re
import time
from groq import Groq
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SETUP & CONFIGURATION ---
# FIX: We MUST load the .env file BEFORE we try to configure the APIs!
load_dotenv()

# --- GEMINI CONFIGURATION & CMO PROMPTS ---
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    genai.configure(api_key=gemini_key)
else:
    # Adding a safety net so the app tells you if your .env file is broken
    st.error("FATAL ERROR: GEMINI_API_KEY is missing from your .env file!")
    st.stop()



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
# --- 2. ASSET LOADING & SEMANTIC EMBEDDINGS ---
# --- 2. ASSET LOADING & LOCAL SEMANTIC EMBEDDINGS ---
# --- 2. ASSET LOADING (ENTERPRISE 3.0 HYBRID DATA) ---
@st.cache_resource
def load_assets():
    model = joblib.load('heal_bridge_bnb_model.joblib')
    encoder = joblib.load('disease_label_encoder.joblib')
    
    with open('master_feature_columns.json', 'r', encoding='utf-8') as f:
        master_cols = json.load(f)
    with open('release_evidences.json', 'r', encoding='utf-8') as f:
        evidences = json.load(f)
        
    # NEW: Load the pure clinical keywords
    with open('clinical_keywords.json', 'r', encoding='utf-8') as f:
        clinical_dict = json.load(f)
        
    vocab_list = []
    for key, value in evidences.items():
        question = value.get("question_en", "")
        if question: vocab_list.append(question)
        
    corpus_text = []
    code_map = []
    raw_text_map = [] # We keep the original questions just for human-readable debug logs
    
    for col in master_cols:
        if col in ['AGE', 'SEX'] or col.startswith('AGE_'): continue
        
        # We embed the PURE CLINICAL KEYWORDS, not the questions
        clean_text = clinical_dict.get(col, "")
        corpus_text.append(clean_text)
        code_map.append(col)
        
        # Keep the original question for the debug UI
        if '_@_' in col:
            base_code, val_code = col.split('_@_')
            base_q = evidences.get(base_code, {}).get('question_en', '')
            val_mean = evidences.get(base_code, {}).get('value_meaning', {}).get(val_code, {}).get('en', '')
            raw_text_map.append(f"{base_q} -> {val_mean}")
        else:
            raw_text_map.append(evidences.get(col, {}).get('question_en', col))
            
    print("Loading CLINICAL Semantic Model (PubMedBERT)...")
    embedder = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
    corpus_embeddings = embedder.encode(corpus_text)
    
    print("Building BM25 Lexical Index...")
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [doc.lower().split() for doc in corpus_text]
    bm25 = BM25Okapi(tokenized_corpus)
            
    return model, encoder, master_cols, evidences, vocab_list, corpus_text, code_map, corpus_embeddings, embedder, bm25, raw_text_map

# Load everything into memory
bnb_model, label_encoder, master_columns, evidences, symptoms_list, corpus_text, code_map, corpus_embeddings, embedder, bm25, raw_text_map = load_assets()
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
                            
                            transcript = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages if m['role'] != 'system'])
                            
                            # --- TIER 1: LLM Extraction (Groq/Llama-3 for Blazing Speed & JSON Format) ---
                            # --- TIER 1: CLINICAL TRANSLATION (LLM converts layman to medical terms) ---
                            # --- TIER 1: CLINICAL TRANSLATION (LLM converts layman to medical terms) ---
                            extraction_prompt = f"""You are an elite Clinical NLP Scribe. 
                            Review this triage conversation:
                            
                            {transcript}
                            
                            Extract ALL symptoms the patient confirmed and denied. 
                            CRITICAL RULE: You MUST translate their layman descriptions into precise, standard medical terminology.
                            
                            Examples:
                            - "pain when pressing and letting go" -> "rebound tenderness"
                            - "lower right stomach pain" -> "right lower quadrant abdominal pain"
                            - "watery eye" -> "excessive tearing"
                            - "burns when peeing" -> "dysuria"
                            - "lost appetite" -> "anorexia"
                            
                            Output ONLY a JSON object:
                            {{
                                "present": ["list", "of", "translated", "clinical", "terms"],
                                "absent": ["list", "of", "denied", "clinical", "terms"]
                            }}"""
                            
                            try:
                                extraction = client.chat.completions.create(
                                    model="llama-3.3-70b-versatile",
                                    messages=[{"role": "system", "content": extraction_prompt}],
                                    temperature=0.0,
                                    response_format={"type": "json_object"}
                                )
                                full_json = json.loads(extraction.choices[0].message.content)
                                present_strings = full_json.get("present", [])
                                absent_strings = full_json.get("absent", [])
                                
                                # --- TIER 2: BATCH AGENTIC RAG (70B LLM logically maps the translated terms) ---
                                # --- TIER 2: SEQUENTIAL AGENTIC RAG (Zero Dropped Keys, Rate-Limit Safe) ---
                                valid_json = {}
                                debug_mapping = {}
                                
                                # Combine present and absent into one list for the loop
                                all_terms = [(t, 'present') for t in present_strings] + [(t, 'absent') for t in absent_strings]
                                
                                for term, status in all_terms:
                                    # 1. Hybrid Search for this ONE specific term
                                    tokenized_query = term.lower().split()
                                    bm25_scores = bm25.get_scores(tokenized_query)
                                    if np.max(bm25_scores) > 0:
                                        bm25_scores = bm25_scores / np.max(bm25_scores)
                                        
                                    vec = embedder.encode([term])
                                    sem_scores = cosine_similarity(vec, corpus_embeddings)[0]
                                    
                                    combined_scores = (bm25_scores * 0.4) + (sem_scores * 0.6)
                                    top_indices = np.argsort(combined_scores)[::-1][:7] # Pull top 7 options
                                    
                                    candidates = {code_map[idx]: raw_text_map[idx] for idx in top_indices}
                                    
                                    # 2. Ask 70B to hyper-focus on mapping JUST THIS ONE TERM
                                    prompt = f"""You are a strict clinical mapping auditor.
                                    Target term: "{term}"
                                    
                                    Map this term to the single best DDXPlus code from these options:
                                    {json.dumps(candidates, indent=2)}
                                    
                                    RULES:
                                    1. "rebound tenderness" MUST map to E_144.
                                    2. Location pain (RLQ, abdominal) MUST map to E_55_... codes (NOT E_133 which is for rashes/lesions).
                                    3. If there is no logical match, output "NONE".
                                    
                                    Output ONLY a JSON object: {{"code": "E_XXX"}} or {{"code": "NONE"}}"""
                                    
                                    try:
                                        resp = client.chat.completions.create(
                                            model="llama-3.3-70b-versatile",
                                            messages=[{"role": "system", "content": prompt}],
                                            temperature=0.0,
                                            response_format={"type": "json_object"}
                                        )
                                        result = json.loads(resp.choices[0].message.content)
                                        code = result.get("code")
                                        
                                        if code and code != "NONE" and code in master_columns:
                                            if status == 'present':
                                                valid_json[code] = 1
                                                debug_mapping[f"✅ {term}"] = f"Code: {code} ({candidates.get(code, '')})"
                                            else:
                                                valid_json[code] = 0
                                                debug_mapping[f"❌ {term}"] = f"Code: {code} ({candidates.get(code, '')})"
                                    except:
                                        pass
                                        
                                    # 3. THE MAGIC FIX: Sleep for 1.5 seconds to bypass API rate limits
                                    time.sleep(1.5)

                                # --- TIER 3: HIERARCHICAL BUBBLING & MASTER NODES ---
                                for key in list(valid_json.keys()):
                                    if '_@_' in key and valid_json[key] == 1:
                                        base_key = key.split('_@_')[0]
                                        if base_key in master_columns:
                                            valid_json[base_key] = 1
                                            debug_mapping[f"⬆️ BUBBLE OVERRIDE"] = f"Forced Parent {base_key} to 1 based on {key}"
                                            
                                pain_codes = [k for k in valid_json.keys() if k.startswith('E_54') or k.startswith('E_55') or k.startswith('E_56') or k.startswith('E_57')]
                                if len(pain_codes) > 0 and 'E_53' in master_columns:
                                    valid_json['E_53'] = 1
                                    debug_mapping["🚨 MASTER PAIN OVERRIDE"] = "Forced E_53 (Master Pain Node) to 1."

                                st.session_state.debug_extracted_json = {"1_Translation": full_json, "2_70B_Mapping": debug_mapping}
                                
                                # 3. Initialize Tracker and Run Math
                                # 3. Initialize Tracker and Run Math
                                pat_age = st.session_state.patient_record["age"] if st.session_state.patient_record["age"] is not None else 30
                                pat_sex = st.session_state.patient_record["sex"] if st.session_state.patient_record["sex"] is not None else "male"
                                
                                is_male = True if pat_sex.lower() == "male" else False
                                tracker = PatientTracker(age=pat_age, sex_is_male=is_male)
                                tracker.update_symptoms(valid_json)
                                
                                # --- TIER 4: THE PURE MATH PROTOCOL ---
                                tracker.state[tracker.state == -1] = 0 # Required: The model expects 800 zeros
                                
                                # 1. Use the mathematically perfect built-in Scikit-Learn function
                                raw_probs = bnb_model.predict_proba(tracker.state)[0]
                                
                                # 2. Safely filter out impossible diseases POST-prediction
                                for i, disease in enumerate(label_encoder.classes_):
                                    # Nuke exotic diseases for standard clinics
                                    if disease in ['Ebola', 'Chagas', 'Larygospasm']:
                                        raw_probs[i] *= 0.00001
                                    # Age-gate pediatric diseases so adults don't get Croup
                                    elif disease in ['Croup', 'Bronchiolitis', 'Whooping cough'] and pat_age > 15:
                                        raw_probs[i] *= 0.00001
                                        
                                # 3. Re-normalize probabilities to 100%
                                probs = raw_probs / np.sum(raw_probs)
                                top_3_indices = np.argsort(probs)[::-1][:3]
                                # 4. Save the Results to Memory!
                                st.session_state.top_3_results = []
                                top_3_names = []
                                for i, idx in enumerate(top_3_indices):
                                    disease = label_encoder.classes_[idx]
                                    top_3_names.append(disease)
                                    confidence = probs[idx] * 100
                                    st.session_state.top_3_results.append(f"**{i+1}. {disease}** ({confidence:.1f}% confidence)")
                                
                                # 5. THE CHIEF MEDICAL OFFICER (Sanity Check & Dual Report)
                                cmo_eval_prompt = f"""You are the Chief Medical Officer, the ultimate clinical safety net.
                                
                                PATIENT DATA:
                                - Age: {pat_age}
                                - Sex: {pat_sex}
                                - Confirmed Symptoms: {present_strings}
                                - Denied Symptoms: {absent_strings}
                                
                                MATH ENGINE PREDICTIONS: {top_3_names}
                                
                                YOUR MISSION:
                                1. SANITY CHECK: Does the Math Engine's #1 prediction make logical clinical sense? (e.g., if it predicted Ebola for a headache, or COPD for a 2-year-old, it is critically wrong).
                                2. OVERRIDE: If the engine is wrong, you MUST override it with the correct medical diagnosis based purely on the symptoms provided.
                                3. DUAL REPORT: Write two separate reports based on the final verified diagnosis. One for the patient, and one for the attending doctor.
                                
                                Output ONLY a JSON object in this exact format:
                                {{
                                    "engine_was_correct": true/false,
                                    "final_diagnosis": "The verified disease name",
                                    "patient_friendly_report": "Speak directly to the patient. Use zero medical jargon. Be empathetic, explain the condition simply, and provide actionable next steps.",
                                    "clinical_report": "Speak to the attending physician. Use strict medical terminology. Summarize the clinical presentation, why this diagnosis fits, and recommended clinical management."
                                }}"""
                                
                                try:
                                    cmo_resp = client.chat.completions.create(
                                        model="llama-3.3-70b-versatile",
                                        messages=[{"role": "system", "content": cmo_eval_prompt}],
                                        temperature=0.0, # 0.0 forces strict logical reasoning
                                        response_format={"type": "json_object"}
                                    )
                                    cmo_evaluation = json.loads(cmo_resp.choices[0].message.content)
                                    
                                    st.session_state.cmo_eval = cmo_evaluation
                                    st.session_state.triage_complete = True
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"CMO Evaluation Error: {e}")
                                
                            except Exception as e:
                                st.error(f"Diagnostic Engine Error: {e}")
                    
                    # 5. THE NORMAL CHAT FLOW (If we aren't diagnosing yet)
                    # 5. THE NORMAL CHAT FLOW (If we aren't diagnosing yet)
                    else:
                        try:
                            # Using 70B for much better conversational empathy and logic
                            chat_completion = client.chat.completions.create(
                                model="llama-3.3-70b-versatile", # <--- SWAPPED TO 70B
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


# --- 7. FINAL DIAGNOSIS SCREEN & CMO DUAL REPORT ---
if st.session_state.triage_complete:
    st.success("✅ Assessment Complete. Your report has been generated.")
    
    cmo_data = st.session_state.cmo_eval
    
    # DUAL REPORT TABS (Debug logs, Overrides, and explicit Disease Names are hidden)
    tab1, tab2 = st.tabs(["👤 Patient Report", "⚕️ Clinical Report (Medical)"])
    
    with tab1:
        st.write(cmo_data.get("patient_friendly_report", ""))
        
    with tab2:
        st.write(cmo_data.get("clinical_report", ""))
        
    st.markdown("---")

    # 4. Action Chips & Follow-Up (We reset the chat history here to start fresh with the CMO)

    # 4. Action Chips & Follow-Up (We reset the chat history here to start fresh with the CMO)
    if "cmo_messages" not in st.session_state:
        st.session_state.cmo_messages = []
        
    st.write("💡 **Suggested Questions:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💊 What are my treatment options?", use_container_width=True):
            st.session_state.chip_trigger = "What are my treatment options?"
            st.rerun()
    with col2:
        if st.button("🍲 Are there any home remedies?", use_container_width=True):
            st.session_state.chip_trigger = "Are there any home remedies?"
            st.rerun()
    with col3:
        # NEW: Dos and Don'ts Button
        if st.button("✅❌ Dos and Don'ts", use_container_width=True):
            st.session_state.chip_trigger = "What are the immediate Dos and Don'ts for my condition right now?"
            st.rerun()

    # 5. The Follow-Up Chat Box (Using Groq instead of Gemini to prevent limits)
    cmo_input = st.chat_input("Ask a follow-up question about your diagnosis...")
    
    if st.session_state.get("chip_trigger"):
        cmo_input = st.session_state.chip_trigger
        del st.session_state.chip_trigger  
        
    if cmo_input:
        st.session_state.cmo_messages.append({"role": "user", "content": cmo_input})
        
        for msg in st.session_state.cmo_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        with st.chat_message("assistant"):
            with st.spinner("The Chief Medical Officer is typing..."):
                try:
                    # THE ELITE CMO FOLLOW-UP GUARDRAILS
                    # THE ELITE CMO FOLLOW-UP GUARDRAILS (EXTREME BREVITY)
                    # THE ELITE CMO FOLLOW-UP GUARDRAILS (BALANCED & SCANNABLE)
                    cmo_follow_up_rules = f"""You are an elite, empathetic clinical AI discussing a triage diagnosis of {cmo_data.get('final_diagnosis')}.
                    
                    CRITICAL RULES FOR YOUR RESPONSE:
                    1. THE GOLDILOCKS LENGTH: Do not write a massive essay, but do not be a robot. Give clear, 1 to 2 sentence explanations for every point you make.
                    2. SCANNABLE FORMAT: Use bolded bullet points so the patient can read it quickly.
                    3. ZERO FLUFF: NEVER use introductory filler like "As your Chief Medical Officer" or "I am glad you asked." Start delivering the medical information immediately.
                    4. CLINICAL CAUTION: For surgical or life-threatening emergencies (Appendicitis, Heart Attack, Stroke, etc.):
                       - STRICTLY BAN home remedies, heat, or ice.
                       - EXPLICITLY WARN the patient NOT to eat, drink, or take pain meds.
                    5. THE DOCTOR PIVOT: End your response with exactly one clear, professional sentence directing them to urgent care or a human doctor for official treatment.
                    """
                    
                    follow_up_prompt = [{"role": "system", "content": cmo_follow_up_rules}]
                    follow_up_prompt.extend(st.session_state.cmo_messages)
                    
                    cmo_response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=follow_up_prompt,
                        temperature=0.3 # Lowered temperature to stop it from "getting creative" with remedies
                    )
                    reply_text = cmo_response.choices[0].message.content
                    st.markdown(reply_text)
                    st.session_state.cmo_messages.append({"role": "assistant", "content": reply_text})
                except Exception as e:
                    st.error(f"CMO Connection Error: {e}")