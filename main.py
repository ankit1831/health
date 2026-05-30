import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from fastapi.responses import StreamingResponse # 🟢 NEW IMPORT
import pandas as pd 
import json
import joblib
import time
import re
from typing import Optional
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import uvicorn

# 1. SETUP
load_dotenv()
client = Groq()
app = FastAPI()

# 2. LOAD ASSETS
model = joblib.load('heal_bridge_bnb_model.joblib')
encoder = joblib.load('disease_label_encoder.joblib')
with open('master_feature_columns.json', 'r', encoding='utf-8') as f:
    master_cols = json.load(f)
with open('release_evidences.json', 'r', encoding='utf-8') as f:
    evidences = json.load(f)
with open('clinical_keywords.json', 'r', encoding='utf-8') as f:
    clinical_dict = json.load(f)

corpus_text = []
code_map = []
for col in master_cols:
    if col in ['AGE', 'SEX'] or col.startswith('AGE_'): continue
    corpus_text.append(clinical_dict.get(col, ""))
    code_map.append(col)

embedder = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
corpus_embeddings = embedder.encode(corpus_text)

# --- TRIAGE SEVERITY DICT FOR THE CMO ---
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
    primary_disease = top_3_diseases[0] if top_3_diseases else ""
    severity = TRIAGE_SEVERITY.get(primary_disease, "YELLOW")
    if severity == "RED": return "RED", "[SYSTEM ALERT: CRITICAL EMERGENCY. Instruct the patient to go to the nearest ER. DO NOT suggest home remedies.]"
    elif severity == "YELLOW": return "YELLOW", "[SYSTEM ALERT: URGENT CONDITION. Instruct the patient to schedule an appointment with a doctor soon. Provide safe temporary management.]"
    else: return "GREEN", "[SYSTEM ALERT: MINOR CONDITION. Reassure the patient. Provide safe over-the-counter home care.]"

# 3. API DATA MODELS
class TriageRequest(BaseModel):
    age: Optional[int] = 30 
    sex: Optional[str] = "male" 
    transcript: str

class ChatRequest(BaseModel):
    message: Optional[str] = ""
    history: Optional[list] = []
    age: Optional[int] = None
    sex: Optional[str] = None

class FollowUpRequest(BaseModel):
    diagnosis: str
    question: str
    history: str

# 4. PRE-TRIAGE CONVERSATION ENDPOINT
@app.post("/api/chat")
def run_chat(req: ChatRequest):
    try:
        safe_message = req.message if req.message else ""
        safe_history = req.history if req.history else []
        
        extracted_age = req.age
        extracted_sex = req.sex
        
       # 1. INDESTRUCTIBLE DEMOGRAPHIC EXTRACTION
        # Combine the current message and all past history into one searchable string
        full_context = safe_message.lower() + " " + " ".join([m.get("content", "").lower() for m in safe_history])
        
        if not extracted_age:
            age_match = re.search(r'\b(\d{1,3})\b', full_context)
            if age_match: extracted_age = int(age_match.group(1))
            
        if not extracted_sex:
            # Catch strict matches, common typos, and natural synonyms
            sex_match = re.search(r'\b(male|female|femlae|woman|man|boy|girl)\b', full_context)
            if sex_match: 
                matched_word = sex_match.group(1)
                extracted_sex = "Female" if matched_word in ["female", "femlae", "woman", "girl"] else "Male"

        # 2. SECTOR COUNT
        receptionist_asks = sum(1 for msg in safe_history if msg.get("role") == "assistant" and ("age" in msg.get("content", "").lower() or "sex" in msg.get("content", "").lower()))

        # 3. Check if we ALREADY used the fallback
        has_advanced = any("proceed without those exact details" in msg.get("content", "") for msg in safe_history)

        # 4. FALLBACK OVERRIDE
        if (not extracted_age or not extracted_sex) and receptionist_asks >= 2 and not has_advanced:
            reply = "No problem at all. We can proceed without those exact details. Please describe the main symptoms you are experiencing today."
            # NEW: Explicitly returning skip_demographics so the UI knows to show the button
            return {"reply": reply, "extracted_age": extracted_age, "extracted_sex": extracted_sex, "skip_demographics": True}

        # 5. SELECT THE PERSONA
        if (not extracted_age or not extracted_sex) and not has_advanced:
            system_prompt = """You are a strict clinical receptionist. Your ONLY job is to ask for the patient's age and biological sex. 
            CRITICAL RULES:
            1. DO NOT ask about symptoms yet.
            2. Be extremely concise. Maximum 2 sentences.
            3. DO NOT give medical advice or excessive sympathy."""
        else:
            system_prompt = """You are an elite Clinical Triage Diagnostician. 
            YOUR BEHAVIOR:
            1. ONE QUESTION MAXIMUM: You must NEVER ask more than one question in a single message.
            2. NO FLUFF: Be direct and clinical. Do not use excessive sympathy.
            3. DIFFERENTIAL DIAGNOSIS: Ask highly specific questions that differentiate between similar diseases. 
            4. FLEXIBILITY: If the patient says they want to talk more, explore more symptoms, or refuses to predict, you MUST oblige and continue asking diagnostic questions without arguing."""

        messages = [{"role": "system", "content": system_prompt}] + safe_history
        messages.append({"role": "user", "content": safe_message})
        
        # 🟢 NEW: Tell Groq to stream the output
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.4,
            stream=True  # <--- CRITICAL: Enables chunking
        )
        
        # 🟢 NEW: Generator function for Server-Sent Events (SSE)
        def generate():
            # 1. Send the metadata instantly so the UI can update the profile/buttons
            metadata = {
                "type": "metadata",
                "extracted_age": extracted_age,
                "extracted_sex": extracted_sex,
                "skip_demographics": has_advanced
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            
            # 2. Check if we need to send the hardcoded demographic confirmation
            if (not req.age or not req.sex) and (extracted_age and extracted_sex):
                hardcoded_reply = f"Thank you. I have recorded your details: {extracted_age} years old, {extracted_sex}. Please describe the main symptoms you are experiencing today."
                yield f"data: {json.dumps({'type': 'chunk', 'text': hardcoded_reply})}\n\n"
                return # Stop here so it doesn't stream the actual AI prompt

            # 3. Stream the live Llama-3 text chunks word-by-word
            for chunk in resp:
                if chunk.choices[0].delta.content is not None:
                    text_chunk = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'type': 'chunk', 'text': text_chunk})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        print(f"\n🚨 BACKEND CRASH IN /api/chat 🚨\nError Details: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))

# 5. THE TRIAGE ENDPOINT
@app.post("/api/triage")
def run_triage(req: TriageRequest):
    try:
        # Protect mathematical runtime with solid fallbacks
        final_age = req.age if req.age is not None else 30
        final_sex = req.sex if req.sex is not None else "male"
             
        # --- TIER 1: CLINICAL TRANSLATION ---
        extraction_prompt = f"""You are an elite Clinical NLP Scribe. 
        Review this triage conversation:\n{req.transcript}\n
        Extract ALL symptoms the patient confirmed and denied. Translate to medical terminology.
        Output ONLY a JSON object: {{"present": [], "absent": []}}"""
        
        extraction = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": extraction_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        full_json = json.loads(extraction.choices[0].message.content)
        
        # ==========================================
        # 🟢 FIX BUG 2: THE "ZERO SYMPTOM" INTERCEPTOR 🟢
        # ==========================================
        if not full_json.get("present") and not full_json.get("absent"):
            raise HTTPException(
                status_code=400, 
                detail="No clear medical symptoms were detected in the chat. Please describe what you are feeling (e.g., 'I have a headache' or 'My stomach hurts') before running the diagnosis."
            )
        # ==========================================

        # --- TIER 2: SEQUENTIAL MAPPING ---
        valid_json = {}
        all_terms = [(t, 'present') for t in full_json.get("present", [])] + [(t, 'absent') for t in full_json.get("absent", [])]
        
        for term, status in all_terms:
            vec = embedder.encode([term])
            sem_scores = np.dot(vec, corpus_embeddings.T)[0]
            top_indices = np.argsort(sem_scores)[::-1][:7]
            candidates = {code_map[idx]: corpus_text[idx] for idx in top_indices}
            
            prompt = f"""You are a strict clinical mapping auditor. Target term: "{term}"
            Map this term to the single best DDXPlus code from these options:\n{json.dumps(candidates, indent=2)}\n
            RULES:
            1. "rebound tenderness" MUST map to E_144.
            2. Location pain (RLQ, abdominal) MUST map to E_55_... codes.
            3. If there is no logical match, output "NONE".
            Output ONLY a JSON object: {{"code": "E_XXX"}} or {{"code": "NONE"}}"""
            
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            code = json.loads(resp.choices[0].message.content).get("code")
            if code and code != "NONE" and code in master_cols:
                valid_json[code] = 1 if status == 'present' else 0
            time.sleep(0.5) 

        # --- TIER 4: PURE MATH PROTOCOL ---
        state_df = pd.DataFrame(np.zeros((1, len(master_cols)), dtype=np.int8), columns=master_cols)
        
        if final_age <= 18: state_df.at[0, 'AGE_0_18'] = 1
        elif final_age <= 35: state_df.at[0, 'AGE_19_35'] = 1
        elif final_age <= 50: state_df.at[0, 'AGE_36_50'] = 1
        elif final_age <= 65: state_df.at[0, 'AGE_51_65'] = 1
        else: state_df.at[0, 'AGE_66_PLUS'] = 1
        
        state_df.at[0, 'SEX'] = 1 if final_sex.lower() == "male" else 0
        
        for code, value in valid_json.items():
            if code in state_df.columns:
                state_df.at[0, code] = value

        state_df = state_df.reindex(columns=model.feature_names_in_, fill_value=0)
        raw_probs = model.predict_proba(state_df)[0]
        
        for i, disease in enumerate(encoder.classes_):
            if disease in ['Ebola', 'Chagas', 'Larygospasm', 'HIV (initial infection)', 'Guillain-Barré syndrome', 'Myasthenia gravis', 'Acute dystonic reactions', 'Tuberculosis']:
                raw_probs[i] *= 0.000001
            elif disease in ['Acute COPD exacerbation / infection', 'Possible NSTEMI / STEMI', 'Pulmonary neoplasm'] and final_age < 30:
                raw_probs[i] *= 0.000001
            elif disease in ['Croup', 'Bronchiolitis', 'Whooping cough']:
                raw_probs[i] *= 0.000001 if final_age > 15 else 50.0
            elif disease in ['URTI', 'Viral pharyngitis', 'Influenza', 'Bronchitis', 'Acute rhinosinusitis', 'GERD', 'Cluster headache', 'Tension-type headache', 'Panic attack', 'Acute otitis media']:
                raw_probs[i] *= 100.0

        probs = raw_probs / np.sum(raw_probs)
        top_3_names = [encoder.classes_[idx] for idx in np.argsort(probs)[::-1][:3]]

        # --- TIER 5: THE CHIEF MEDICAL OFFICER ---
        severity, alert = evaluate_clinical_severity(top_3_names)
        
        cmo_eval_prompt = f"""You are the Chief Medical Officer, an expert, professional clinical AI assistant.
        PATIENT DATA: Age: {final_age}, Sex: {final_sex}
        SYMPTOMS: {full_json.get('present')}
        
        THE DIAGNOSTIC ENGINE'S PREDICTION: {', '.join(top_3_names)}. 
        🚨 CRITICAL SAFETY OVERRIDE: {alert}
        
        YOUR RULES FOR THE PATIENT REPORT:
        1. TONE: Be professional, direct, and supportive. DO NOT use overly emotional apologies (e.g., avoid "I am so sorry you are experiencing this" or "I can imagine how overwhelming this is").
        2. EXPLANATION: Explain the final diagnosis in simple, plain English. What is happening in their body?
        3. ACTIONABLE: Clearly state the severity of the condition and tell them exactly what to do next (e.g., go to the ER immediately, schedule a doctor's appointment, or rest at home).
        4. CRITICAL: NEVER mention "the diagnostic engine" or AI probabilities. Own the diagnosis. Do not recommend specific prescription drug dosages.
        
        Output ONLY a JSON object:
        {{
            "engine_was_correct": true/false,
            "final_diagnosis": "The final verified disease name",
            "patient_friendly_report": "Write a concise, 2-paragraph explanation following the rules above. Keep it factual but reassuring.",
            "clinical_report": "Write a detailed 1-paragraph medical summary for the attending physician."
        }}"""
        
        cmo_resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": cmo_eval_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        # Intercept response to inject the programmatic triage severity tag for front-end rendering
        output_payload = json.loads(cmo_resp.choices[0].message.content)
        output_payload["severity"] = severity 
        
        # 🟢 NEW: Pass the extracted symptoms down for the PDF Export
        output_payload["extracted_symptoms"] = full_json.get('present', []) 
        
        return output_payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 6. THE FOLLOW UP CHAT
@app.post("/api/followup")
def run_followup(req: FollowUpRequest):
    try:
        cmo_follow_up_rules = f"""You are a professional, helpful clinical AI discussing a triage diagnosis of {req.diagnosis}.
        CRITICAL RULES:
        1. CONVERSATIONAL TONE: Speak naturally and directly. DO NOT use overly emotional or dramatic sympathy (e.g., no "I'm so sorry to hear that"). Be reassuring but factual.
        2. NO DRUG PRESCRIPTIONS: Do NOT recommend specific prescription drug names or dosages.
        3. EXPLAIN CLEARLY: Answer their question simply and accurately.
        4. THE DOCTOR PIVOT: Always gently remind them to consult a human doctor for an official treatment plan."""
        
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": cmo_follow_up_rules},
                {"role": "user", "content": f"Previous context:\n{req.history}\n\nPatient Question: {req.question}"}
            ],
            temperature=0.3
        )
        return {"reply": resp.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 7. SERVE THE HTML FRONTEND
app.mount("/static", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
def read_index():
    return FileResponse("frontend/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)