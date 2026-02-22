import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Paste your Ground Truth paragraph from the terminal here!
TEST_TEXT = """ I am an 49-year-old female. I came into the clinic today because Regarding the question 'Did you lose consciousness', my answer is I would describe the pain as a cramp and sharp. The pain is specifically located in my. iliac fossa(R), iliac fossa(L), hypochondrium(R), hypochondrium(L), and epigastric. On a scale of 1 to 10, the intensity is 3. On a scale of 1 to 10, the precision of the pain location is a 4. Regarding how fast the pain appeared, on a scale of 1 to 10, it was a 2. Regarding the question 'What color is the rash', my answer is pink. No, I am not ir lesions peel off.. The affected region is located on my back of the neck, biceps(L), mouth, thyroid cartilage, and ankle(R). On a scale of 1 to 10, the intensity is 3. Yes, regarding the question 'is the lesion (or are the lesions) larger than 1cm', my answer is. On a scale of 1 to 10, the severity of the itching is a 10. The swelling is located on my forehead, cheek(R), and cheek(L). No, I am not traveling out of the country recently. To give you more context: I have a known severe food allergy. Regarding the question 'Have you been in contact with or ate something that you have an allergy to', my answer is Regarding the question 'Have you had diarrhea or an increase in stool frequency', my answer is I have pain somewhere, related to your reason for consulting. I am experiencing shortness of breath or difficulty breathing in a significant way. I have any lesions, redness or problems on your skin that you believe are related to the condition you are consulting for. I am feeling nauseous or do you feel like vomiting. I have swelling in one or more areas of your body. Regarding the question 'Have you noticed a high pitched sound when breathing in', my answer is I am more likely to develop common allergies than the general population."""

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

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=49, problem_type="multi_label_classification")
model.load_state_dict(torch.load("saved_models/heal_bridge_best_model.pt", map_location='cpu')['model_state_dict'])
model.eval()

print("Analyzing...")
encoding = tokenizer(TEST_TEXT, add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
with torch.no_grad():
    outputs = model(**encoding)
    probs = torch.sigmoid(outputs.logits)[0]
    
top_probs, top_indices = torch.topk(probs, 3)
for i in range(3):
    print(f"{i+1}. {DISEASES[top_indices[i].item()]} ({top_probs[i].item()*100:.2f}%)")