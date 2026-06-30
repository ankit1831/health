import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
import os
import base64
from pydantic import BaseModel
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
import uvicorn
import requests

# 1. SETUP
load_dotenv()
client = Groq()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains (perfect for portfolios)
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, etc.
    allow_headers=["*"],
)
# Configure Gemini API Key

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

import requests # Make sure this is imported at the top!

# --- SERVERLESS AGENTIC RAG SETUP ---
# --- SERVERLESS AGENTIC RAG SETUP ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/pritamdeka/S-PubMedBert-MS-MARCO"

def get_hf_embeddings(texts):
    """Fetches embeddings from Hugging Face with a bulletproof network fallback"""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    try:
        # Added a 15-second timeout so it doesn't hang forever if the network is bad
        response = requests.post(API_URL, headers=headers, json={"inputs": texts}, timeout=15)
        response.raise_for_status() # Instantly triggers the except block if it gets a 503 or 404
        return np.array(response.json())
        
    except Exception as e:
        print(f"⚠️ Network or HF API Error: {str(e)}")
        print("⚠️ Falling back to a safe zero-matrix to prevent server crash.")
        # Returns an empty matrix so the server successfully boots even without internet
        return np.zeros((len(texts), 768)) 

print("🔄 Connecting to Hugging Face Inference API...")
corpus_embeddings = get_hf_embeddings(corpus_text)
print("✅ Agentic RAG Vector Space Initialized!")

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

class DocumentPayload(BaseModel):
    mime_type: str
    data: str
    prompt: str
    phase: str = "triage" # 🟢 Required for the CMO phase check to work

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

        # 4. FALLBACK OVERRIDE FLAG (Do not return a dict here!)
        trigger_fallback = (not extracted_age or not extracted_sex) and receptionist_asks >= 2 and not has_advanced

        # 5. SELECT THE PERSONA
        if not trigger_fallback and (not extracted_age or not extracted_sex) and not has_advanced:
            system_prompt = """
            CRITICAL: YOU MUST MATCH THE PATIENT'S LANGUAGE SCRIPT. 
                - If the patient's latest message is in English, you MUST reply in English.
                - If the patient's latest message is in Hindi (Devanagari script) or Hinglish (e.g., "bukhar ਹੈ", "sir dard"), you MUST reply in Hindi script.
                - If unsure, default to English.
        You are a strict clinical receptionist. Your ONLY job is to ask for the patient's age and biological sex. 
            CRITICAL RULES:
            1. DO NOT ask about symptoms yet.
            2. Be extremely concise. Maximum 2 sentences.
            3. DO NOT give medical advice or excessive sympathy.
            """
        else:
            system_prompt = """
            CRITICAL: YOU MUST MATCH THE PATIENT'S LANGUAGE SCRIPT. 
            - If the patient's latest message is in English, you MUST reply in English.
            - If the patient's latest message is in Hindi (Devanagari script) or Hinglish (e.g., "bukhar ਹੈ", "sir dard"), you MUST reply in Hindi script.
            - If unsure, default to English.
            You are an elite Clinical Triage Diagnostician. 
            YOUR BEHAVIOR:
            1. ONE QUESTION MAXIMUM: You must NEVER ask more than one question in a single message.
            2. NO FLUFF: Be direct and clinical. Do not use excessive sympathy.
            3. DIFFERENTIAL DIAGNOSIS: Ask highly specific questions that differentiate between similar diseases. 
            4. 🟢 ABSOLUTE VISUAL PRIORITY: If a [System Image Scan Results] or [System Document Analysis] block is present in the transcript history, you MUST drop any historical focus on generic text symptoms (like a simple headache) and dedicate your questions entirely to exploring the findings of that visual image scan. It is the primary chief complaint.
            LANGUAGE PROTOCOL: 
            1. Analyze the language of the patient's MOST RECENT message.
            2. If the message is in English (even with typos), you MUST reply in English.
            3. ONLY if the message contains obvious Hindi script (Devanagari) OR clear Hindi phrases written in English (Hinglish like "mujhe bukhar hai"), you must reply in Hindi.
            4. If you are unsure, DEFAULT TO ENGLISH."""
        messages = [{"role": "system", "content": system_prompt}] + safe_history
        messages.append({"role": "user", "content": safe_message})
        
        # 🟢 NEW: Tell Groq to stream the outputfa
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.4,
            stream=True  # <--- CRITICAL: Enables chunking
        )
        
        # 🟢 NEW: Generator function for Server-Sent Events (SSE)
        def generate():
            # 🟢 THE CRASH FIX: Handle the fallback safely inside the stream
            if trigger_fallback:
                metadata = {"type": "metadata", "extracted_age": extracted_age, "extracted_sex": extracted_sex, "skip_demographics": True}
                yield f"data: {json.dumps(metadata)}\n\n"
                
                reply = "No problem at all. We can proceed without those exact details. Please describe the main symptoms you are experiencing today."
                yield f"data: {json.dumps({'type': 'chunk', 'text': reply})}\n\n"
                return

            # 1. Send the normal metadata instantly
            metadata = {
                "type": "metadata",
                "extracted_age": extracted_age,
                "extracted_sex": extracted_sex,
                "skip_demographics": has_advanced
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            
            # 2. Check if we need to send the hardcoded demographic confirmation
            if (not req.age or not req.sex) and (extracted_age and extracted_sex):
                base_reply = f"Thank you. I have recorded your details: {extracted_age} years old, {extracted_sex}. "
                
                # Convert the incoming request object to a string to search the chat history
                chat_context = str(req)
                
                # 1. Did the user upload ANY document?
                if "successfully analyzed your document" in chat_context:
                    
                    # 2. Does the chat history contain an ABNORMAL document?
                    if "specific parameters" in chat_context:
                        final_reply = base_reply + "I see you have already uploaded a document with notable metrics. You can click 'Predict' to run the diagnosis, or tell me any other physical symptoms you feel."
                    else:
                        final_reply = base_reply + "I see your uploaded document was completely normal. Please describe any physical discomfort or symptoms you are experiencing so we can investigate further."
                        
                # 4. Standard Flow (No documents uploaded at all)
                else:
                    final_reply = base_reply + "Please describe the main symptoms you are experiencing today."
                    
                yield f"data: {json.dumps({'type': 'chunk', 'text': final_reply})}\n\n"
                return # 740715 INDENTED INSIDE THE IF BLOCK

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
        CRITICAL RULES:
        1. You MUST extract any medical parameters, conditions, or symptoms found inside the [System Document Analysis: ...] blocks as 'present' symptoms. Treat these as absolute facts.
        2. 🔴 EXCEPTION OVERRIDE: If the patient replies with "no", "incorrect", or explicitly denies the accuracy of the document right after the AI asks "Does this extraction look accurate?", you MUST completely IGNORE the [System Document Analysis] block. Do NOT extract those visual parameters.
        3. ONLY extract physical symptoms the patient explicitly claims to feel. Do not assume symptoms just because the AI asked about them.

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
            vec = get_hf_embeddings([term])
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
        CRITICAL LANGUAGE DIRECTIVE:
        - Scan the RAW PATIENT CHAT HISTORY above.
        - If the patient typed in Hindi or Hinglish, you MUST write the "patient_friendly_report" entirely in Hindi (Devanagari script).
        - If the patient typed in English, write the "patient_friendly_report" in English.
        - The "clinical_report" field MUST always stay in English for the medical professional.
        
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
            "severity": You MUST evaluate the danger level and output ONLY ONE of these exact words: "RED" (Life-threatening/ER), "YELLOW" (Urgent/Doctor within 48h), or "GREEN" (Minor/Home care).
            "patient_friendly_report": "Write a concise, 2-paragraph explanation following the rules above. Keep it factual but reassuring. And keep it in simple terms that a patient can easily understand. Avoid medical jargon.",
            "clinical_report": "Write a detailed 2-paragraph medical summary for the attending physician."

        }}"""
        
        cmo_resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": cmo_eval_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        # Intercept response to inject the programmatic triage severity tag for front-end rendering
        output_payload = json.loads(cmo_resp.choices[0].message.content)
        
        # 🟢 THE FIX: Delete the old overwrite, and enforce the LLM's dynamic severity
        # We use .get() to pull the severity Llama-3 chose, defaulting to GREEN if it glitches
        output_payload["severity"] = output_payload.get("severity", "GREEN").upper()
        
        # Pass the extracted symptoms down for the PDF Export
        output_payload["extracted_symptoms"] = full_json.get('present', []) 
        
        return output_payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/hospitals")
def get_hospitals(lat: float, lon: float):
    overpass_url = "https://overpass-api.de/api/interpreter"
    # This is the exact query you were using
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"](around:25000,{lat},{lon});
      way["amenity"="hospital"](around:25000,{lat},{lon});
      node["healthcare"="hospital"](around:25000,{lat},{lon});
    );
    out center;
    """
    try:
        response = requests.post(overpass_url, data=query, headers={"User-Agent": "HealBridge-App/1.0"})
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/analyze-document")
async def analyze_document(payload: DocumentPayload):
    try:
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY is missing from environment variables.")

        # 🟢 Initialize the NEW Google GenAI Client
        client = genai.Client() 
        
        # 🟢 Use the NEW SDK format for handling image bytes
        raw_bytes = base64.b64decode(payload.data)
        file_blob = types.Part.from_bytes(data=raw_bytes, mime_type=payload.mime_type)
        
        # CMO PHASE
        # CMO PHASE
        if payload.phase == "cmo":
            cmo_prompt = f"""
            CRITICAL: YOU MUST MATCH THE LANGUAGE OF THE PATIENT'S PROMPT.
            - If the patient's prompt is in English, reply in English.
            - If the patient's prompt is in Hindi or Hinglish, reply entirely in Hindi (Devanagari script).
            You are a Chief Medical Officer. The patient uploaded this image and asked: '{payload.prompt}'. 
            Answer clinically and briefly.
            """
            
            # Use the fallback array to bypass 503 server overloads
            models_to_try = ['gemini-2.5-flash', 'gemini-3.1-flash-lite', 'gemini-3.5-flash', 'gemini-2.5-flash-lite']
            cmo_reply = None
            
            for model_name in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[cmo_prompt, file_blob]
                    )
                    cmo_reply = response.text.strip()
                    break # Success, exit the loop
                except Exception as e:
                    print(f"⚠️ CMO Model {model_name} failed. Trying next...")
                    continue
                    
            if not cmo_reply:
                # If ALL models fail, return a safe UI message instead of a 500 crash
                return {
                    "extracted_symptoms": "CMO_VISION_ANALYSIS",
                    "ai_reply": "⏳ **System Cooldown:** The medical vision engine is currently at maximum capacity. Please wait a moment and try asking your question again.",
                    "age": None, "sex": None
                }
                
            return {
                "extracted_symptoms": "CMO_VISION_ANALYSIS",
                "ai_reply": cmo_reply,
                "age": None, "sex": None
            }
            
        system_prompt = """
        You are an elite clinical data verification and parsing system. Analyze the provided image or file.
        CRITICAL RULE: NEVER include the words "STEP 1", "STEP 2", or "STEP 3" in your response. Output ONLY the clean data.
        
        STEP 1: VALIDATION
        Evaluate if this file is a medical document, lab test report, prescription, clinical summary, or an image showing a physiological symptom.
        - If the file is completely unrelated to medicine, healthcare, or physiology, reply with EXACTLY the token: INVALID_MEDICAL_FILE
        
        STEP 2: CLINICAL ANALYSIS
        If valid, review the markers, numbers, or visual features.
        - Case A (Abnormalities Found): List only the out-of-range, elevated, low, or clinically relevant abnormal metrics/symptoms as a clean, comma-separated list.
        - Case B (All Clear / Completely Normal): Provide a structured summary of the key metrics read in this exact format:
          NORMAL_REPORT_BREAKDOWN: [Metric Name] is [Value] [Unit] (Reference Range: [Range]) | [Next Metric Name]...

        STEP 3: AGE AND BIOMETRIC EXTRACTION
        Scan the document for the patient's Age and Sex. If not explicitly written, use UNKNOWN.  
        FORMAT EXACTLY LIKE THIS:
        AGE: [Age or UNKNOWN]
        SEX: [Sex or UNKNOWN]
        """

        models_to_try = [
            'gemini-2.5-flash',       
            'gemini-3.1-flash-lite',  
            'gemini-3.5-flash',       
            'gemini-2.5-flash-lite'   
        ]
        
        extracted_text = None
        
        for model_name in models_to_try:
            try:
                print(f"🔄 Attempting inference with {model_name}...")
                
                # 🟢 Generate Content using the new Client architecture
                response = client.models.generate_content(
                    model=model_name,
                    contents=[system_prompt, file_blob, payload.prompt]
                )
                
                extracted_text = response.text.strip()
                print(f"✅ Success with {model_name}!")
                break
                
            except Exception as e:
                error_str = str(e).lower()
                # 🟢 FIX: Added 503 and unavailable so it triggers the fallback loop!
                if "429" in error_str or "503" in error_str or "unavailable" in error_str or "quota" in error_str or "exhausted" in error_str or "not found" in error_str:
                    print(f"⚠️ {model_name} failed (Quota/Missing). Routing to next model...")
                    continue 
                else:
                    raise e
                    
        # If the loop finishes and we STILL don't have text, EVERY model failed.
        if not extracted_text:
            return {
                "extracted_symptoms": "System rate limits triggered across all models.",
                "ai_reply": "⏳ **System Cooldown:** The medical vision engine is currently processing maximum capacity. Please wait a moment and try sending this document again."
            }

        # --- DYNAMIC INTERACTION ROUTING (Using the successful extracted_text) ---
        
        # 🟢 MINIMAL ADDITION 1: Parse Age and Sex before routing
        lines = extracted_text.split('\n')
        age_val = "UNKNOWN"
        sex_val = "UNKNOWN"
        clean_metrics = extracted_text
        
        for line in lines:
            if line.startswith("AGE:"): 
                age_val = line.replace("AGE:", "").strip()
                clean_metrics = clean_metrics.replace(line, "").strip() # Remove from metrics string
            if line.startswith("SEX:"): 
                sex_val = line.replace("SEX:", "").strip()
                clean_metrics = clean_metrics.replace(line, "").strip() # Remove from metrics string

        # 🟢 MINIMAL ADDITION 2: Update Returns to include Age, Sex, and the "Verify" question
        if "INVALID_MEDICAL_FILE" in extracted_text:
            return {
                "extracted_symptoms": "Invalid document uploaded.",
                "ai_reply": "⚠️ **Invalid Upload:** The file provided does not appear to be a medical report, clinical document, or physical symptom image. Please upload a valid health document for analysis.",
                "age": None, "sex": None
            }
            
        elif "NORMAL_REPORT_BREAKDOWN:" in clean_metrics:
            breakdown_data = clean_metrics.replace("NORMAL_REPORT_BREAKDOWN:", "").strip()
            formatted_list = ""
            for item in breakdown_data.split("|"):
                if item.strip():
                    formatted_list += f"\n• {item.strip()}"
            
            # Add demographics and verification to the normal report
            natural_reply = "I have successfully analyzed your document.\n\n"
            if age_val != "UNKNOWN":
                natural_reply += f"**Demographics Found:** {age_val} years old, {sex_val}.\n\n"
            
            natural_reply += (
                f"All tested markers look excellent and fall safely inside normal clinical boundaries:\n"
                f"{formatted_list}\n\n"
                f"**Does this extraction look accurate to you?**" # Verification ask
            )
            
            return {
                "extracted_symptoms": "No abnormalities detected. Document confirmed healthy.",
                "ai_reply": natural_reply,
                "age": age_val, 
                "sex": sex_val
            }
            
        else:
            # Add demographics and verification to the abnormal report
            natural_reply = "I have successfully analyzed your document.\n\n"
            if age_val != "UNKNOWN":
                natural_reply += f"**Demographics Found:** {age_val} years old, {sex_val}.\n\n"
            
            natural_reply += (
                f"I noted the following specific parameters: *{clean_metrics}*.\n\n"
                f"**Does this extraction look accurate to you?**" # Verification ask
            )
            
            return {
                "extracted_symptoms": clean_metrics,
                "ai_reply": natural_reply,
                "age": age_val, 
                "sex": sex_val
            }

    except Exception as e:
        print(f"❌ BACKEND FATAL ERROR: {str(e)}") 
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/followup")
def run_followup(req: FollowUpRequest):
    try:
        cmo_follow_up_rules = f"""
        CRITICAL: YOU MUST MATCH THE LANGUAGE OF THE PATIENT'S QUESTION.
        - If the patient asks their question in English, reply in English.
        - If the patient asks their question in Hindi or Hinglish, reply entirely in Hindi (Devanagari script).
        You are a professional, helpful clinical AI discussing a triage diagnosis of {req.diagnosis}.
        CRITICAL RULES:
        1. CONVERSATIONAL TONE: Speak naturally and directly. DO NOT use overly emotional or dramatic sympathy (e.g., no "I'm so sorry to hear that"). Be reassuring but factual.
        2. NO DRUG PRESCRIPTIONS: Do NOT recommend specific prescription drug names or dosages.
        3. EXPLAIN CLEARLY: Answer their question simply and accurately.
        4. THE DOCTOR PIVOT: Always gently remind them to consult a human doctor for an official treatment plan.
        5. USE BULLET POINTS: Break down instructions, symptoms, or remedies into clear, scannable bullet points or paragraphs.
        6. USE HIGHLIGHTING: **Bold** key medical terms, medications, and critical warnings.
        7. BE CONCISE: Get straight to the point. Maximum 1-2 short paragraphs/bullet sections.
        """
        
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

# 7. PURE API HEALTH CHECK (Updated for Split-Stack)
@app.get("/")
def read_root():
    return {"status": "Heal Bridge Engine is Live and Operational"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)