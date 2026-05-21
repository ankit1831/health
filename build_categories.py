import json
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq()

print("--- BUILDING THE ANATOMICAL MAP ---")

# 1. Load data
with open('symptom_hierarchy.json', 'r', encoding='utf-8') as f:
    hierarchy = json.load(f)
with open('release_evidences.json', 'r', encoding='utf-8') as f:
    evidences = json.load(f)

# Extract just the Level 1 Parents and their English questions
parents = {code: evidences[code].get('question_en', code) for code in hierarchy.keys()}
parent_items = list(parents.items())

categories_map = {}

# The categories we want the LLM to use
valid_categories = [
    "Neurological", "Respiratory", "Gastrointestinal", "Cardiovascular", 
    "Musculoskeletal", "Dermatological", "Psychiatric", "Ear/Nose/Throat", "Systemic/Other"
]

# 2. Chunking to avoid API Rate Limits
chunk_size = 40
chunks = [parent_items[i:i + chunk_size] for i in range(0, len(parent_items), chunk_size)]

print(f"Divided {len(parents)} symptoms into {len(chunks)} chunks.")

for i, chunk in enumerate(chunks):
    print(f"Processing Chunk {i+1}/{len(chunks)}...")
    
    # Format the chunk for the prompt
    chunk_dict = {code: text for code, text in chunk}
    
    prompt = f"""You are a Chief Medical Officer categorizing symptoms. 
    Map each of the following symptom codes to exactly ONE of these categories:
    {json.dumps(valid_categories)}
    
    SYMPTOMS TO CATEGORIZE:
    {json.dumps(chunk_dict, indent=2)}
    
    Output ONLY a valid JSON object where keys are the symptom codes (e.g., "E_91") and values are the exact category name. Do not output anything else."""
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(completion.choices[0].message.content)
        categories_map.update(result)
        
        # Pause for 3 seconds between chunks to respect the API limits
        time.sleep(3)
        
    except Exception as e:
        print(f"Error on chunk {i+1}: {e}")

# 3. Save the brain map
with open('symptom_categories.json', 'w', encoding='utf-8') as f:
    json.dump(categories_map, f, indent=4)

print("\n✅ Anatomical Map successfully built and saved to 'symptom_categories.json'!")
print(f"Total Categorized: {len(categories_map)}/{len(parents)}")