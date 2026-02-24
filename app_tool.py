import streamlit as st
from groq import Groq
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json
import re
import time
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
        for key, value in data.items():
            question = value.get("question_en", "")
            if question: symptoms.append(question)
        return symptoms[:100] # Give the LLM plenty of options
    except FileNotFoundError:
        return ["pain", "cough", "fever", "shortness of breath", "rash"]

symptoms_list = load_ddxplus_vocabulary()

# --- THE CLEAN BEHAVIORAL PROMPT ---
# --- THE IRONCLAD BEHAVIORAL PROMPT ---
# --- THE IRONCLAD BEHAVIORAL PROMPT ---
# --- THE IRONCLAD BEHAVIORAL PROMPT ---
SYSTEM_PROMPT = f"""
You are an elite Clinical Triage Diagnostician. 

YOUR BEHAVIOR (STRICT RULES):
1. ONE QUESTION MAXIMUM: You must NEVER ask more than one question in a single message.
2. DYNAMIC QUESTIONING: Based on the chief complaint, ask targeted questions from this allowed vocabulary list: {', '.join(symptoms_list[:80])}
3. KEEP ASKING: Keep asking questions one by one. Do not stop until the user explicitly says "stop" or "predict now".

DATA EXTRACTION RULES (WHEN CALLING THE TOOL):
- If the user answered YES to a symptom, put the exact symptom string in the `positive_symptoms` array.
- If the user answered NO to a symptom, put the exact symptom string in the `negative_symptoms` array.
- NEVER write negative statements (e.g., "no pain") in the positive array. Just put "pain" in the negative array.
"""
# --- DEFINE THE PYTHON TOOL FOR GROQ (WITH ENUMS) ---
# --- DEFINE THE PYTHON TOOL FOR GROQ (SOFT CONSTRAINTS) ---
triage_tool = {
    "type": "function",
    "function": {
        "name": "submit_triage_assessment",
        "description": "CRITICAL: Call this tool ONLY if the user explicitly types 'predict now', 'stop', OR if you have already asked at least 5 targeted follow-up questions in the conversation. DO NOT call this tool immediately after the user's first message.",
        "parameters": {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "description": "Patient's age in years. Use 0 if unknown."},
                "sex": {
                    "type": "string", 
                    "description": "Patient's sex ('male' or 'female'). Use 'unknown' if not provided."
                },
                "chief_complaint": {
                    "type": "string", 
                    "description": "The symptom from the allowed vocabulary for their main issue."
                },
                "positive_symptoms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Symptoms the user explicitly confirmed (said YES to)."
                },
                "negative_symptoms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Symptoms the user explicitly denied (said NO to). If none, use an empty array []."
                }
            },
            # Notice we removed "age" and "sex" from this list!
            "required": ["chief_complaint", "positive_symptoms", "negative_symptoms"]
        }
    }
}
# --- THE RECEPTIONIST TOOL ---
demographics_tool = {
    "type": "function",
    "function": {
        "name": "save_demographics",
        "description": "Call this tool IMMEDIATELY once the user provides both their age and sex.",
        "parameters": {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "description": "Patient's age in years."},
                "sex": {"type": "string", "description": "Patient's sex (male or female)."}
            },
            "required": ["age", "sex"]
        }
    }
}
@st.cache_resource 
def load_ai_engine():
    device = torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModelForSequenceClassification.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT", num_labels=49, problem_type="multi_label_classification"
    )
    checkpoint = torch.load("saved_models/heal_bridge_best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return tokenizer, model, device

# --- INITIALIZE SESSION ---
if "groq_client" not in st.session_state:
    st.session_state.groq_client = Groq()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello. I am the Heal Bridge Triage Assistant. Could you please tell me your age, sex, and what brings you in today?"}
    ]
    st.session_state.triage_complete = False 

if "patient_record" not in st.session_state:
    st.session_state.patient_record = {"age": None, "sex": None}
    
if "demo_ask_count" not in st.session_state:
    st.session_state.demo_ask_count = 0

if "next_intercept_turn" not in st.session_state:
    st.session_state.next_intercept_turn = 6

st.title("🩺 Heal Bridge Intake")
# --- THE UI ESCAPE HATCH ---
if "force_predict" not in st.session_state:
    st.session_state.force_predict = False

with st.sidebar:
    st.markdown("### Patient Controls")
    if st.button("🩺 Generate Diagnosis Now", use_container_width=True):
        st.session_state.force_predict = True
        st.rerun()


for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if not st.session_state.triage_complete:
    # 1. Calculate if they have hit the rolling limit
    user_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
    limit_reached = (user_turns >= st.session_state.next_intercept_turn)
    
    # 2. INTERCEPT: Show buttons every time they hit the target turn
    if limit_reached:
        st.info("💡 I have gathered a detailed clinical picture. Would you like to proceed to the diagnostic results, or do you have more symptoms to share?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🩺 Generate Diagnosis", use_container_width=True):
                st.session_state.force_predict = True
                # Push the target to an impossible number so buttons don't flash while loading
                st.session_state.next_intercept_turn += 100 
                st.rerun()
        with col2:
            if st.button("💬 Add More Symptoms", use_container_width=True):
                # Move the target exactly 5 turns into the future!
                st.session_state.next_intercept_turn += 5
                st.rerun()
                
    # 3. NORMAL FLOW: Show the chat box if we aren't intercepting
    else:
        user_input = st.chat_input("Type your symptoms here...")
        
        # Trigger if user types something OR if they click the sidebar/intercept button
        if user_input or st.session_state.force_predict:
            
            # Inject the hidden stop command if they clicked the diagnosis button
            if st.session_state.force_predict and not user_input:
                user_input = "predict now"
                st.session_state.messages.append({"role": "user", "content": "I am ready for my diagnosis now."})
                with st.chat_message("user"):
                    st.markdown("*User requested immediate diagnosis.*")
            else:
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                    
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        # ... (Your existing try block and State Router Logic stays EXACTLY the same below this) ...
                    # --- STATE ROUTER LOGIC ---
                    
                        is_missing_demographics = (st.session_state.patient_record["age"] is None)
                    
                    # 1. Check if the user typed a stop word manually
                        user_typed_stop = any(trigger in user_input.lower() for trigger in ["predict now", "stop", "enough"])
                    
                    # 2. Count the user's total turns
                        user_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
                    
                    # 3. SMART TRIGGERS (Word Counts)
                        total_user_words = sum(len(m["content"].split()) for m in st.session_state.messages if m["role"] == "user")
                        latest_msg_words = len(user_input.split())
                    
                    # --- THE TRIPLE THREAT EVALUATION ---
                        is_bailout = (
                            user_typed_stop or 
                            st.session_state.force_predict or 
                            user_turns >= 20 or 
                            (latest_msg_words >= 40 and not is_missing_demographics) # They dumped a massive paragraph just now
                               # They have typed enough overall
                        )

                    # Reset the UI button state
                        st.session_state.force_predict = False
                        
                        # 1. CHECK FOR FALLBACK FIRST (Did they ignore us 3 times?)
                        if is_missing_demographics and not is_bailout:
                            
                            # MINIMAL FIX: Check if the user is trying to type their age or sex right now
                            # MINIMAL FIX: Check for digits or exact word matches for gender
                            user_text = user_input.lower()
                            is_typing_demographics = any(char.isdigit() for char in user_text) or bool(re.search(r'\b(male|female)\b', user_text))
                            
                            # Only force defaults if they hit the limit AND they aren't typing their age!
                            if st.session_state.demo_ask_count >= 3 and not is_typing_demographics:
                                # They ignored us 3 times. Force defaults and move to Triage.
                                st.session_state.patient_record["age"] = 30
                                st.session_state.patient_record["sex"] = "male"
                                is_missing_demographics = False
                        
                        # 2. THE SOFT GATE (Attempting to get Demographics)
                        # 2. THE SOFT GATE (Attempting to get Demographics)
                        if is_missing_demographics and not is_bailout:
                            st.session_state.demo_ask_count += 1
                            
                            dynamic_system_prompt = """You are a medical receptionist. Your ONLY job is to get the patient's age and sex. 
                            If they mention symptoms, express empathy, but gently insist you need their age and sex first. 
                            DO NOT ask diagnostic questions."""
                            
                            api_args = {
                                "messages": [{"role": "system", "content": dynamic_system_prompt}] + st.session_state.messages,
                                "model": "llama-3.1-8b-instant", #llama-3.3-70b-versatile
                                "temperature": 0.2,
                                "tools": [demographics_tool],
                                "tool_choice": "auto"
                            }
                        
                        # 3. THE TRIAGE NURSE (We have the data, or we forced the defaults)
                        # 3. THE TRIAGE NURSE (We have the data, or we forced the defaults)
                        else:
                            dynamic_system_prompt = SYSTEM_PROMPT 
                            
                            api_args = {
                                # Change [1:] to just st.session_state.messages here too!
                                "messages": [{"role": "system", "content": dynamic_system_prompt}] + st.session_state.messages,
                                "model": "llama-3.1-8b-instant",#llama-3.1-8b-instant
                                "temperature": 0.0,
                            }
                            
                        
                            if is_bailout:
                                api_args["tools"] = [triage_tool]
                                # CRITICAL FIX: Do not give the AI a choice. FORCE it to run the tool.
                                api_args["tool_choice"] = {"type": "function", "function": {"name": "submit_triage_assessment"}}
                                
                            # Reset the button state
                            st.session_state.force_predict = False

                        # --- SEND TO GROQ (WITH RATE LIMIT ARMOR) ---
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                chat_completion = st.session_state.groq_client.chat.completions.create(**api_args)
                                response_message = chat_completion.choices[0].message
                                break  # If successful, break out of the retry loop immediately
                            except Exception as e:
                                error_str = str(e).lower()
                                # If it's a rate limit error and we have retries left, wait and try again
                                if "429" in error_str or "rate limit" in error_str:
                                    if attempt < max_retries - 1:
                                        time.sleep(4)  # Silently pause for 4 seconds
                                        continue
                                # If it's a different error, or we ran out of retries, crash gracefully
                                raise e
                        
                        # --- INTERCEPT TOOL CALLS ---
                        if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                            tool_call = response_message.tool_calls[0]
                            tool_name = tool_call.function.name
                            data = json.loads(tool_call.function.arguments)
                            
                            # Handle Receptionist Tool
                            if tool_name == "save_demographics":
                                st.session_state.patient_record["age"] = data.get("age")
                                st.session_state.patient_record["sex"] = data.get("sex")
                                
                                transition_msg = "Thank you. I have recorded your details. Let's talk about your symptoms. Can you tell me more about what you are experiencing?"
                                st.markdown(transition_msg)
                                st.session_state.messages.append({"role": "assistant", "content": transition_msg})
                                st.rerun()
                                
                            # Handle Triage Tool (The PyTorch Engine)
                            elif tool_name == "submit_triage_assessment":
                                st.session_state.triage_complete = True
                                # Use the locked-in demographics, not what the LLM guessed
                                patient_age = st.session_state.patient_record["age"]
                                patient_sex = st.session_state.patient_record["sex"]
                                
                                def clean_text(text):
                                    if not text: return ""
                                    text = re.sub(r'^(Do you have|Are you experiencing|Do you feel like you are \(or were\)|Do you feel like you are|Have you had) ', '', text, flags=re.IGNORECASE)
                                    return text.replace('?', '').replace('your ', 'my ').replace('any ', '').strip().lower()

                                cc_clean = clean_text(data.get('chief_complaint', 'pain'))
                                synthetic_text = f"I am an {patient_age}-year-old {patient_sex}. I came into the clinic today because I have {cc_clean}. "
                                
                                for symp in data.get('positive_symptoms', []):
                                    if symp: synthetic_text += f"To give you more context: I have {clean_text(symp)}. "
                                for neg in data.get('negative_symptoms', []):
                                    if neg: synthetic_text += f"I do not have {clean_text(neg)}. "
                                    
                                with st.spinner("Analyzing..."):
                                    tokenizer, torch_model, device = load_ai_engine()
                                    encoding = tokenizer(synthetic_text, add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
                                    with torch.no_grad():
                                        outputs = torch_model(input_ids=encoding['input_ids'].to(device), attention_mask=encoding['attention_mask'].to(device))
                                        probs = torch.sigmoid(outputs.logits)[0]
                                    top_k_probs, top_k_indices = torch.topk(probs, 3)
                                    
                                    report = f"### 🩺 Diagnostic Predictions:\n**Synthesized Data:**\n> *{synthetic_text}*\n\n"
                                    for i in range(3): report += f"**{i+1}. {DISEASES[top_k_indices[i].item()]}** ({top_k_probs[i].item()*100:.2f}%)\n"
                                    
                                st.session_state.messages.append({"role": "assistant", "content": report})
                                st.rerun()
                                
                        # --- NORMAL CHAT BEHAVIOR ---
                        else:
                            ai_reply = response_message.content.strip()
                            st.markdown(ai_reply)
                            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                            
                    except Exception as e:
                        st.error(f"API Error: {e}")