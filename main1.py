import joblib
import json
import numpy as np
import pandas as pd

print("Loading ML Assets into Memory...")
# 1. LOAD THE NEW ASSETS (Make sure these 3 files are in your backend folder)
model = joblib.load('heal_bridge_bnb_model.joblib')
encoder = joblib.load('disease_label_encoder.joblib')

with open('master_feature_columns.json', 'r', encoding='utf-8') as f:
    master_cols = json.load(f)

# Optional: Load clinical keywords if your backend does the NLP mapping here
# with open('clinical_keywords.json', 'r', encoding='utf-8') as f:
#     clinical_dict = json.load(f)

def predict_diagnosis(age: int, sex: str, extracted_symptom_codes: list):
    """
    Takes patient details and returns top 5 predictions.
    extracted_symptom_codes should be a list like: ['E_55_@_V_29', 'E_201']
    """
    
    # 1. CREATE THE ZERO-ARRAY (Using Pandas for named columns)
    state_df = pd.DataFrame(np.zeros((1, len(master_cols)), dtype=np.int8), columns=master_cols)
    
    # 2. APPLY AGE BINS LOGIC
    if age <= 18:
        state_df.at[0, 'AGE_0_18'] = 1
    elif age <= 35:
        state_df.at[0, 'AGE_19_35'] = 1
    elif age <= 50:
        state_df.at[0, 'AGE_36_50'] = 1
    elif age <= 65:
        state_df.at[0, 'AGE_51_65'] = 1
    else:
        state_df.at[0, 'AGE_66_PLUS'] = 1
        
    # 3. APPLY SEX LOGIC (1 for Male, 0 for Female)
    state_df.at[0, 'SEX'] = 1 if sex.upper() == 'M' else 0
    
    # 4. FLIP EXTRACTED SYMPTOMS TO 1
    for code in extracted_symptom_codes:
        if code in state_df.columns:
            state_df.at[0, code] = 1
            
    # 5. THE MAGIC FIX: FORCE PERFECT SEQUENCE ALIGNMENT
    # This guarantees the columns are in the exact order the Model's brain expects.
    state_df = state_df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # 6. GET PREDICTIONS
    raw_probs = model.predict_proba(state_df)[0]
    top_indices = np.argsort(raw_probs)[::-1][:5] # Get Top 5
    
    results = []
    for idx in top_indices:
        prob = float(raw_probs[idx]) * 100
        if prob > 0.01: # Only return mathematically relevant results
            results.append({
                "disease": str(encoder.classes_[idx]),
                "probability": round(prob, 2)
            })
            
    return results

# ==========================================
# QUICK TEST TO VERIFY IT WORKS IN YOUR BACKEND
# ==========================================
if __name__ == "__main__":
    print("\nTesting the Backend Pipeline...")
    # Simulating a 45-year-old male with Chest Pain extracted by your NLP
    simulated_codes = ['E_55_@_V_29', 'E_55_@_V_56'] 
    
    output = predict_diagnosis(age=45, sex='M', extracted_symptom_codes=simulated_codes)
    
    print("\n🔥 API JSON RESPONSE:")
    print(json.dumps(output, indent=4))