import streamlit as st
from groq import Groq
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json
import re
import time
from dotenv import load_dotenv
import google.generativeai as genai
import os

# 1. Configure Gemini (Assuming load_dotenv() is already called at the top of your app)
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    genai.configure(api_key=gemini_key)

# 2. The Triage Safety Net (Clinical Severity Mapping)
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

# 3. The Safety Override Logic
def evaluate_clinical_severity(top_3_diseases):
    severities = [TRIAGE_SEVERITY.get(disease, "YELLOW") for disease in top_3_diseases]
    if "RED" in severities:
        return "RED", "[SYSTEM ALERT: CRITICAL EMERGENCY. Instruct the patient to go to the nearest Emergency Room or call an ambulance immediately. DO NOT suggest home remedies. Keep the tone urgent but calm.]"
    elif "YELLOW" in severities:
        return "YELLOW", "[SYSTEM ALERT: URGENT CONDITION. Instruct the patient to schedule an appointment with a doctor soon. Provide safe, temporary symptom management but emphasize medical evaluation.]"
    else:
        return "GREEN", "[SYSTEM ALERT: MINOR CONDITION. Reassure the patient. Provide safe, over-the-counter home care management and lifestyle adjustments.]"

def generate_cmo_prompt(top_3_diseases, system_alert):
    top_disease = top_3_diseases[0]
    prompt = f"""You are the Chief Medical Officer, a highly empathetic, professional, and expert clinical AI assistant.
The PyTorch Triage Engine has analyzed the patient's symptoms. The top suspected conditions are: {', '.join(top_3_diseases)}. The primary focus is: {top_disease}.

🚨 CRITICAL SAFETY OVERRIDE:
{system_alert}

YOUR BEHAVIORAL RULES:
1. THE TONE: Speak directly to the patient with deep empathy and a warm, clinical bedside manner. 
2. NO MATH: DO NOT show the patient the raw list of diseases or any percentages. 
3. PLAIN ENGLISH: Explain what {top_disease} is in very simple, non-medical terms.
4. NO PRESCRIPTIONS: NEVER recommend specific drug dosages or brand-name medications. 
5. THE HANDOFF: Strictly follow the CRITICAL SAFETY OVERRIDE.
6. BE CONCISE (CRITICAL): The patient is unwell and tired. Keep your paragraphs extremely short (1-2 sentences maximum). Do not write long preambles.
7. SCANNABILITY: If providing steps or advice, use a MAXIMUM of 3 to 4 short bullet points. Bold the key words so it is easy to read.
8. THE CONVERSATION LOOP: End your message by asking if they have more questions, but VARY your phrasing naturally. Do not repeat the exact same sentence every time.
⚠️ SYMPTOM ESCALATION PROTOCOL (CRITICAL RULE) ⚠️
You are ONLY authorized to discuss {top_disease}. 
If the patient mentions ANY NEW SYMPTOMS during this chat that were not part of their original complaint (especially severe symptoms like spreading redness, chest pain, difficulty breathing, or high fever), YOU MUST IMMEDIATELY STOP giving home care advice.
You must say exactly this: "You are describing new symptoms that change your clinical picture. I cannot safely advise on this condition anymore. Please refresh the page to restart the Triage Assessment with these new symptoms, or seek immediate in-person medical care."

"""
    return prompt

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
                    "description": "Symptoms the user explicitly denied (said NO to). If none, use an empty array [].DO NOT list unasked symptoms. Maximum 5 items."
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
                                    
                                    # ✅ PASTE THIS PHASE 5 BLOCK ✅
                                        # 1. Extract exact disease names into a list for Gemini
                                    top_3_predictions = [DISEASES[top_k_indices[i].item()] for i in range(3)]
                                    
                                    if "cmo_chat" not in st.session_state:
                                        # 2. Evaluate the safety (Red/Yellow/Green)
                                        severity, alert = evaluate_clinical_severity(top_3_predictions)
                                        
                                        # 3. Build the Gemini Model
                                        cmo_model = genai.GenerativeModel(
                                            model_name="gemini-2.5-flash", 
                                            system_instruction=generate_cmo_prompt(top_3_predictions, alert)
                                        )
                                        
                                        # 4. Start the chat and lock it in Streamlit's memory
                                        st.session_state.cmo_chat = cmo_model.start_chat(history=[])
                                        
                                        # 5. Generate the very first Final Report automatically
                                        initial_response = st.session_state.cmo_chat.send_message("Please give me my triage results.")
                                        
                                        # 6. Create a NEW list to store the CMO chat history for the UI
                                        st.session_state.cmo_messages = [{"role": "assistant", "content": initial_response.text}]
                                        
                                        # 7. Change the app's state to move to the final Q&A screen
                                        st.session_state.current_phase = "CMO_QA" 
                                        st.rerun()
                                
                        # --- NORMAL CHAT BEHAVIOR ---
                        else:
                            ai_reply = response_message.content.strip()
                            st.markdown(ai_reply)
                            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                            
                    except Exception as e:
                        st.error(f"API Error: {e}")

# --- PHASE 5 UI: THE CHIEF MEDICAL OFFICER Q&A ---

# This only triggers after PyTorch is done and the state changes
if st.session_state.get("triage_complete") and st.session_state.get("current_phase") == "CMO_QA":
    
    st.success("Triage Assessment Complete. Transferring to the Chief Medical Officer...")
    
    # 1. Draw the CMO Conversation History
    for msg in st.session_state.cmo_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # 2. DYNAMIC ACTION CHIPS (Suggested Questions)
    # Only show these if the AI just spoke, so they disappear while typing/loading
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
            if st.button("⚠️ Is this condition dangerous?", use_container_width=True):
                st.session_state.chip_trigger = "Is this condition dangerous?"
                st.rerun()

    # 3. The New Follow-Up Chat Box
    cmo_input = st.chat_input("Ask a follow-up question about your results...")
    
    # Check if a button was clicked OR if the user typed something manually
    if st.session_state.get("chip_trigger"):
        cmo_input = st.session_state.chip_trigger
        del st.session_state.chip_trigger  # Clear the trigger so it doesn't loop forever
        
    if cmo_input:
        # Append and display the user's question
        st.session_state.cmo_messages.append({"role": "user", "content": cmo_input})
        with st.chat_message("user"):
            st.markdown(cmo_input)
            
        # Get Gemini's response contextually
        with st.chat_message("assistant"):
            with st.spinner("The Chief Medical Officer is typing..."):
                try:
                    cmo_response = st.session_state.cmo_chat.send_message(cmo_input)
                    st.markdown(cmo_response.text)
                    st.session_state.cmo_messages.append({"role": "assistant", "content": cmo_response.text})
                    st.rerun() # Rerun to refresh the action chips!
                except Exception as e:
                    st.error(f"CMO Connection Error: {e}")