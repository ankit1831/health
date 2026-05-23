import json
import time
from groq import Groq
from dotenv import load_dotenv

# Load API Keys
load_dotenv()
client = Groq()

print("Loading raw DDXPlus data...")
with open('release_evidences.json', 'r', encoding='utf-8') as f:
    evidences = json.load(f)
with open('master_feature_columns.json', 'r', encoding='utf-8') as f:
    master_cols = json.load(f)

# 1. Build the raw conversational dictionary
raw_dict = {}
for col in master_cols:
    if col in ['AGE', 'SEX'] or col.startswith('AGE_'): continue
    if '_@_' in col:
        base_code, val_code = col.split('_@_')
        base_q = evidences.get(base_code, {}).get('question_en', '')
        val_mean = evidences.get(base_code, {}).get('value_meaning', {}).get(val_code, {}).get('en', '')
        raw_dict[col] = f"{base_q} -> {val_mean}"
    else:
        raw_dict[col] = evidences.get(col, {}).get('question_en', col)

# 2. Translate in batches to avoid LLM Rate Limits & Hallucinations
clinical_dict = {}
keys = list(raw_dict.keys())
batch_size = 40  # Small batches to ensure perfect JSON output

print(f"Translating {len(keys)} symptoms into clinical keywords using Llama 3...")

for i in range(0, len(keys), batch_size):
    batch_keys = keys[i:i+batch_size]
    batch_data = {k: raw_dict[k] for k in batch_keys}
    
    prompt = f"""You are a strict medical dictionary translator.
    Convert the following conversational medical questions into a concise list of 3 to 5 raw clinical keywords/synonyms.
    
    Example Input: {{"E_127": "Do you feel that your eyes produce excessive tears?"}}
    Example Output: {{"E_127": "watery eye, excessive tearing, crying, epiphora"}}
    
    Translate this batch:
    {json.dumps(batch_data, indent=2)}
    
    Output ONLY a JSON object mapping the exact keys to the new clinical keywords string. Do NOT use markdown."""
    
    success = False
    while not success:
        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(completion.choices[0].message.content)
            clinical_dict.update(result)
            print(f"✅ Processed {len(clinical_dict)} / {len(keys)} codes...")
            success = True
            time.sleep(2) # Breath between API calls
            
        except Exception as e:
            print(f"⚠️ Rate limit or JSON error: {e}. Retrying batch in 5 seconds...")
            time.sleep(5)

# 3. Add a fallback in case the LLM skipped any keys
for k in keys:
    if k not in clinical_dict:
        clinical_dict[k] = raw_dict[k] # Fallback to original if missed

# 4. Save the new enterprise dictionary
with open('clinical_keywords.json', 'w', encoding='utf-8') as f:
    json.dump(clinical_dict, f, indent=4)
    
print("🎉 SUCCESS! clinical_keywords.json has been generated.")