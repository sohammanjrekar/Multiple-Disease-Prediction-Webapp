# import pandas as pd

# data = pd.read_excel(r'C:\Users\HARSH\Desktop\SymptomSolver\antibiotic.xlsx')

# # Convert the dataset into a dictionary grouped by disease
# data_dict = data.set_index('Disease').T.to_dict()

# # Display the resulting dictionary for use in the Flask app
# print(data_dict)

antibiotic_data =  {

'Fungal infection': {
        'Antibiotic': 'Fluconazole',
        'Dosage': '150mg once daily',
        'Duration': '2-4 weeks',
        'Side Effects': 'Nausea, headache',
        'Description': 'Fluconazole is used to treat fungal infections. Adults typically take 150mg once daily for up to 4 weeks.'
    },
    'Allergy': {
        'Antibiotic': 'Loratadine',
        'Dosage': '10mg once daily',
        'Duration': 'As needed',
        'Side Effects': 'Drowsiness, dry mouth',
        'Description': 'Loratadine is an antihistamine used to relieve allergy symptoms. Adults take 10mg once daily as needed.'
    },
    'GERD': {
        'Antibiotic': 'Omeprazole',
        'Dosage': '20mg once daily',
        'Duration': '4-8 weeks',
        'Side Effects': 'Headache, diarrhea',
        'Description': 'Omeprazole is used to treat GERD by reducing stomach acid. Typical dosage is 20mg once daily for 4-8 weeks.'
    },
    'Chronic cholestasis': {
        'Antibiotic': 'Ursodiol',
        'Dosage': '300mg twice daily',
        'Duration': 'Varies',
        'Side Effects': 'Diarrhea, nausea',
        'Description': 'Ursodiol helps improve bile flow in the liver. Dosage is 300mg twice daily; duration varies depending on condition.'
    },
    'Drug Reaction': {
        'Antibiotic': 'Diphenhydramine',
        'Dosage': '25-50mg every 6-8 hours',
        'Duration': 'As needed',
        'Side Effects': 'Drowsiness, dry mouth',
        'Description': 'Diphenhydramine is used to treat allergic reactions. Adults take 25-50mg every 6-8 hours as needed.'
    },
    'Peptic ulcer disease': {
        'Antibiotic': 'Amoxicillin',
        'Dosage': '1g twice daily',
        'Duration': '7-14 days',
        'Side Effects': 'Nausea, diarrhea',
        'Description': 'Amoxicillin is an antibiotic used to treat infections causing ulcers. Adults typically take 1g twice daily for 7-14 days.'
    },
    'AIDS': {
        'Antibiotic': 'Trimethoprim/Sulfamethoxazole',
        'Dosage': '160/800mg once daily',
        'Duration': 'Prophylactic',
        'Side Effects': 'Nausea, rash',
        'Description': 'Trimethoprim/Sulfamethoxazole is used as a prophylactic for AIDS. Dosage is 160/800mg once daily.'
    },
    'Diabetes': {
        'Antibiotic': 'Metformin',
        'Dosage': '500mg twice daily',
        'Duration': 'Ongoing',
        'Side Effects': 'Nausea, stomach upset',
        'Description': 'Metformin is used to control blood sugar in type 2 diabetes. Adults typically take 500mg twice daily.'
    },
    'Gastroenteritis': {
        'Antibiotic': 'Metronidazole',
        'Dosage': '500mg every 8 hours',
        'Duration': '5-7 days',
        'Side Effects': 'Nausea, metallic taste',
        'Description': 'Metronidazole is used to treat gastrointestinal infections. Adults take 500mg every 8 hours for 5-7 days.'
    },
    'Bronchial Asthma': {
        'Antibiotic': 'Levofloxacin',
        'Dosage': '500mg once daily',
        'Duration': '5-7 days',
        'Side Effects': 'Dizziness, headache',
        'Description': 'Levofloxacin is used to treat respiratory infections. Adults take 500mg once daily for 5-7 days.'
    },
    'Hypertension': {
        'Antibiotic': 'Hydrochlorothiazide',
        'Dosage': '25mg once daily',
        'Duration': 'Ongoing',
        'Side Effects': 'Dry mouth, dizziness',
        'Description': 'Hydrochlorothiazide is a diuretic used to treat hypertension. The usual dose is 25mg once daily for ongoing treatment.'
    },
    'Migraine': {
        'Antibiotic': 'Sumatriptan',
        'Dosage': '50-100mg as needed',
        'Duration': 'As needed',
        'Side Effects': 'Drowsiness, nausea',
        'Description': 'Sumatriptan is used to treat migraines. Adults take 50-100mg as needed at the onset of symptoms.'
    },
    'Cervical spondylosis': {
        'Antibiotic': 'Ibuprofen',
        'Dosage': '400-600mg every 8 hours',
        'Duration': 'As needed',
        'Side Effects': 'Stomach upset, dizziness',
        'Description': 'Ibuprofen is used for pain relief in cervical spondylosis. Typical dosage is 400-600mg every 8 hours as needed.'
    },
    'Paralysis (brain hemorrhage)': {
        'Antibiotic': 'Heparin',
        'Dosage': 'Varies based on condition',
        'Duration': 'Under supervision',
        'Side Effects': 'Bleeding risk',
        'Description': 'Heparin is used to prevent blood clots in patients with brain hemorrhage. Dosage varies and should be administered under supervision.'
    },
    'Jaundice': {
        'Antibiotic': 'Ursodiol',
        'Dosage': '300mg twice daily',
        'Duration': 'Varies',
        'Side Effects': 'Nausea, diarrhea',
        'Description': 'Ursodiol helps with bile flow in liver diseases such as jaundice. Typical dosage is 300mg twice daily, duration varies.'
    },
    'Malaria': {
        'Antibiotic': 'Artemether/Lumefantrine',
        'Dosage': '20/120mg twice daily',
        'Duration': '3 days',
        'Side Effects': 'Headache, dizziness',
        'Description': 'Artemether/Lumefantrine is used to treat malaria. Adults take 20/120mg twice daily for 3 days.'
    },
    'Chicken pox': {
        'Antibiotic': 'Acyclovir',
        'Dosage': '800mg 5 times daily',
        'Duration': '7-10 days',
        'Side Effects': 'Nausea, diarrhea',
        'Description': 'Acyclovir is used to treat chicken pox. Adults take 800mg 5 times daily for 7-10 days.'
    },
    'Dengue': {
        'Antibiotic': 'Paracetamol',
        'Dosage': '500mg every 6 hours',
        'Duration': 'As needed',
        'Side Effects': 'Nausea, rash',
        'Description': 'Paracetamol is used to reduce fever in dengue. Adults take 500mg every 6 hours as needed.'
    },
    'Typhoid': {
        'Antibiotic': 'Ceftriaxone',
        'Dosage': '1g once daily',
        'Duration': '7-14 days',
        'Side Effects': 'Nausea, diarrhea',
        'Description': 'Ceftriaxone is an antibiotic used to treat typhoid fever. Adults take 1g once daily for 7-14 days.'
    },
    'Hepatitis A': {
        'Antibiotic': 'Supportive care',
        'Dosage': 'Not applicable',
        'Duration': 'Varies',
        'Side Effects': 'None',
        'Description': 'Hepatitis A is managed with supportive care. There is no specific antibiotic treatment; duration varies.'
    },
    'Hepatitis B': {
        'Antibiotic': 'Entecavir',
        'Dosage': '0.5mg once daily',
        'Duration': 'Long-term',
        'Side Effects': 'Nausea, headache',
        'Description': 'Entecavir is used to treat chronic Hepatitis B. The usual dosage is 0.5mg once daily for long-term management.'
    },
    'Hepatitis C': {
        'Antibiotic': 'Sofosbuvir',
        'Dosage': '400mg once daily',
        'Duration': '12 weeks',
        'Side Effects': 'Fatigue, headache',
        'Description': 'Sofosbuvir is used to treat Hepatitis C. The typical dosage is 400mg once daily for 12 weeks.'
    },
    'Hepatitis D': {
        'Antibiotic': 'Interferon-alpha',
        'Dosage': 'Varies, Under supervision',
        'Duration': 'Under supervision',
        'Side Effects': 'Flu-like symptoms',
        'Description': 'Interferon-alpha is used to treat Hepatitis D. Dosage varies and should be administered under supervision. Flu-like symptoms are common.'
    },
    'Hepatitis E': {
        'Antibiotic': 'Supportive care',
        'Dosage': 'Not applicable',
        'Duration': 'Varies',
        'Side Effects': 'None',
        'Description': 'Hepatitis E is treated with supportive care. There is no specific antibiotic for this condition.'
    },
 
 
  'Alcoholic hepatitis': {
        'Antibiotic': 'Prednisolone',
        'Dosage': '40mg once daily',
        'Duration': '4 weeks',
        'Side Effects': 'Stomach upset, mood changes',
        'Description': 'Prednisolone is used to treat alcoholic hepatitis. The usual dosage is 40mg once daily for 4 weeks.'
    },
    'Tuberculosis': {
        'Antibiotic': 'Rifampin',
        'Dosage': '10mg/kg once daily',
        'Duration': '6-9 months',
        'Side Effects': 'Orange discoloration of bodily fluids',
        'Description': 'Rifampin is used to treat tuberculosis. The dosage is 10mg/kg once daily for a prolonged period of 6-9 months.'
    },
    'Common Cold': {
        'Antibiotic': 'Paracetamol',
        'Dosage': '500mg every 6 hours',
        'Duration': 'As needed',
        'Side Effects': 'Nausea, rash',
        'Description': 'Paracetamol is used to alleviate symptoms of the common cold. Adults typically take 500mg every 6 hours as needed.'
    },
    'Pneumonia': {
        'Antibiotic': 'Amoxicillin',
        'Dosage': '500mg every 8 hours',
        'Duration': '7-10 days',
        'Side Effects': 'Nausea, diarrhea',
        'Description': 'Amoxicillin is used to treat pneumonia. The usual dosage is 500mg every 8 hours for 7-10 days.'
    },
    'Dimorphic hemorrhoids (piles)': {
        'Antibiotic': 'Diosmin',
        'Dosage': '500mg twice daily',
        'Duration': '7-10 days',
        'Side Effects': 'Stomach upset',
        'Description': 'Diosmin is used to treat hemorrhoids. The dosage is 500mg twice daily for 7-10 days.'
    },
    'Heart attack': {
        'Antibiotic': 'Aspirin',
        'Dosage': '81mg once daily',
        'Duration': 'Ongoing',
        'Side Effects': 'Bleeding risk',
        'Description': 'Aspirin is used as a preventive treatment for heart attacks. The usual dosage is 81mg once daily for ongoing prevention.'
    },
    'Varicose veins': {
        'Antibiotic': 'Diosmin',
        'Dosage': '500mg twice daily',
        'Duration': '6-8 weeks',
        'Side Effects': 'Stomach upset',
        'Description': 'Diosmin is used to treat varicose veins. The typical dosage is 500mg twice daily for 6-8 weeks.'
    },
    'Hypothyroidism': {
        'Antibiotic': 'Levothyroxine',
        'Dosage': '25-200mcg once daily',
        'Duration': 'Ongoing',
        'Side Effects': 'Palpitations, weight loss',
        'Description': 'Levothyroxine is used to treat hypothyroidism by replacing thyroid hormones. The dosage varies from 25-200mcg once daily.'
    },
    'Hyperthyroidism': {
        'Antibiotic': 'Methimazole',
        'Dosage': '5-20mg once daily',
        'Duration': 'Ongoing',
        'Side Effects': 'Rash, nausea',
        'Description': 'Methimazole is used to treat hyperthyroidism by inhibiting thyroid hormone production. The usual dosage is 5-20mg once daily.'
    },
    'Hypoglycemia': {
        'Antibiotic': 'Glucose tablets',
        'Dosage': '15-20g as needed',
        'Duration': 'As needed',
        'Side Effects': 'None',
        'Description': 'Glucose tablets are used to rapidly raise blood sugar levels in cases of hypoglycemia. Take 15-20g as needed.'
    },
    'Osteoarthritis': {
        'Antibiotic': 'Celecoxib',
        'Dosage': '200mg once daily',
        'Duration': 'Ongoing',
        'Side Effects': 'Stomach upset, dizziness',
        'Description': 'Celecoxib is used to manage pain and inflammation in osteoarthritis. The usual dosage is 200mg once daily.'
    },
    'Arthritis': {
        'Antibiotic': 'Methotrexate',
        'Dosage': '7.5-25mg weekly',
        'Duration': 'Ongoing',
        'Side Effects': 'Nausea, fatigue',
        'Description': 'Methotrexate is used to treat arthritis by suppressing immune function. The typical dosage is 7.5-25mg weekly.'
    },
    '(vertigo) Paroxysmal Positional Vertigo': {
        'Antibiotic': 'Meclizine',
        'Dosage': '25mg as needed',
        'Duration': 'As needed',
        'Side Effects': 'Drowsiness, dry mouth',
        'Description': 'Meclizine is used to treat vertigo. The dosage is 25mg as needed, depending on symptoms.'
    },
    'Acne': {
        'Antibiotic': 'Clindamycin',
        'Dosage': '1% gel once daily',
        'Duration': 'Ongoing',
        'Side Effects': 'Skin irritation',
        'Description': 'Clindamycin is used to treat acne. The typical dosage is 1% gel once daily, as part of an ongoing treatment.'
    },
    'Urinary tract infection': {
        'Antibiotic': 'Ciprofloxacin',
        'Dosage': '250-500mg every 12 hours',
        'Duration': '3-7 days',
        'Side Effects': 'Nausea, dizziness',
        'Description': 'Ciprofloxacin is used to treat urinary tract infections. The typical dosage is 250-500mg every 12 hours for 3-7 days.'
    },
    'Psoriasis': {
        'Antibiotic': 'Methotrexate',
        'Dosage': '7.5-25mg weekly',
        'Duration': 'Ongoing',
        'Side Effects': 'Nausea, fatigue',
        'Description': 'Methotrexate is used to treat psoriasis by suppressing the immune system. The typical dosage is 7.5-25mg weekly.'
    },
    'Impetigo': {
        'Antibiotic': 'Mupirocin',
        'Dosage': '2% ointment 3 times daily',
        'Duration': '7-10 days',
        'Side Effects': 'Skin irritation',
        'Description': 'Mupirocin is used to treat impetigo, a skin infection. The typical dosage is 2% ointment applied 3 times daily for 7-10 days.'
    }
}