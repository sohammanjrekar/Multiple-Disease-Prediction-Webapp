tests_dict = {
    "Fungal infection": {
        "tests": ["KOH Test", "Fungal Culture"],
        "outcome": "Presence of fungal elements.",
        "definition": {
            "KOH Test": "A test where skin, hair, or nails are treated with potassium hydroxide to detect fungal elements.",
            "Fungal Culture": "A test where the sample is placed in a culture medium to identify fungal species."
        }
    },
    "Allergy": {
        "tests": ["Skin Prick Test", "IgE Blood Test"],
        "outcome": "Raised bumps or elevated IgE levels indicate an allergy.",
        "definition": {
            "Skin Prick Test": "A test where small amounts of allergens are pricked into the skin to check for allergic reactions.",
            "IgE Blood Test": "A test that measures levels of IgE antibodies, which rise in allergic conditions."
        }
    },
    "GERD": {
        "tests": ["pH Monitoring Test", "Endoscopy"],
        "outcome": "Abnormal acid levels or inflammation in the esophagus.",
        "definition": {
            "pH Monitoring Test": "A test that measures acid levels in the esophagus over 24 hours.",
            "Endoscopy": "A procedure using a flexible tube with a camera to visualize the esophagus and stomach."
        }
    },
    "Chronic cholestasis": {
        "tests": ["Liver Function Test", "Ultrasound"],
        "outcome": "Elevated liver enzymes and bile flow obstruction.",
        "definition": {
            "Liver Function Test": "A blood test to check liver enzyme levels and function.",
            "Ultrasound": "An imaging test to visualize the liver and bile ducts."
        }
    },
    "Drug Reaction": {
        "tests": ["Patch Test", "Blood Allergy Test"],
        "outcome": "Identification of drug sensitivity.",
        "definition": {
            "Patch Test": "A skin test where small amounts of drugs are applied to check for allergic reactions.",
            "Blood Allergy Test": "A test to detect drug-specific antibodies in the blood."
        }
    },
    "Peptic ulcer disease": {
        "tests": ["Endoscopy", "Urea Breath Test"],
        "outcome": "Ulcers or H. pylori infection in the stomach.",
        "definition": {
            "Endoscopy": "A flexible tube with a camera to detect ulcers in the stomach lining.",
            "Urea Breath Test": "A test for detecting H. pylori bacteria in the stomach."
        }
    },
    "AIDS": {
        "tests": ["HIV ELISA Test", "Western Blot Test"],
        "outcome": "Detection of HIV antibodies.",
        "definition": {
            "HIV ELISA Test": "A screening blood test for HIV antibodies.",
            "Western Blot Test": "A confirmatory test for detecting HIV proteins."
        }
    },
    "Diabetes": {
        "tests": ["Fasting Blood Sugar", "HbA1c Test"],
        "outcome": "Elevated glucose levels indicate diabetes.",
        "definition": {
            "Fasting Blood Sugar": "A test measuring blood glucose levels after an overnight fast.",
            "HbA1c Test": "A test that provides an average blood sugar level over the past three months."
        }
    },
    "Gastroenteritis": {
        "tests": ["Stool Test", "Electrolyte Test"],
        "outcome": "Presence of infection in stool and electrolyte imbalance.",
        "definition": {
            "Stool Test": "A test to detect bacteria, viruses, or parasites in stool.",
            "Electrolyte Test": "A blood test to check for dehydration and electrolyte imbalance."
        }
    },
    "Bronchial Asthma": {
        "tests": ["Spirometry", "Peak Flow Test"],
        "outcome": "Reduced lung function and airway obstruction.",
        "definition": {
            "Spirometry": "A test that measures lung function and airflow limitation.",
            "Peak Flow Test": "A test measuring how fast a person can breathe out."
        }
    },
    "Hypertension": {
        "tests": ["Blood Pressure Measurement", "Electrocardiogram (ECG)"],
        "outcome": "Elevated blood pressure and heart abnormalities.",
        "definition": {
            "Blood Pressure Measurement": "A test using a sphygmomanometer to measure blood pressure.",
            "Electrocardiogram (ECG)": "A test recording the electrical activity of the heart."
        }
    },
    "Migraine": {
        "tests": ["MRI Scan", "CT Scan"],
        "outcome": "Rule out brain abnormalities.",
        "definition": {
            "MRI Scan": "Magnetic resonance imaging used to detect abnormalities in the brain.",
            "CT Scan": "Computed tomography scan providing detailed brain images to rule out structural issues."
        }
    },
    "Pneumonia": {
        "tests": ["Chest X-ray", "Sputum Culture"],
        "outcome": "Lung inflammation or presence of infection-causing bacteria.",
        "definition": {
            "Chest X-ray": "An imaging test that shows lung infection and inflammation.",
            "Sputum Culture": "A test where mucus from the lungs is analyzed for bacterial infection."
        }
    },
    "Heart attack": {
        "tests": ["ECG", "Troponin Test"],
        "outcome": "Abnormal ECG patterns or elevated troponin levels indicate heart damage.",
        "definition": {
            "ECG": "Electrocardiogram test that records the heart's electrical activity.",
            "Troponin Test": "A blood test measuring levels of troponin, a protein released during heart muscle damage."
        }
    },
    "Tuberculosis": {
        "tests": ["Mantoux Test", "Chest X-ray", "Sputum Test"],
        "outcome": "Presence of TB bacteria in the lungs.",
        "definition": {
            "Mantoux Test": "A skin test to detect latent or active tuberculosis infection.",
            "Chest X-ray": "Used to check for lung abnormalities due to TB.",
            "Sputum Test": "A lab test examining sputum for TB bacteria."
        }
    },
    "Hepatitis A": {
        "tests": ["Liver Function Test", "Hepatitis A Antibody Test"],
        "outcome": "Liver enzyme elevation and presence of hepatitis A antibodies.",
        "definition": {
            "Liver Function Test": "Measures liver enzymes to assess liver health.",
            "Hepatitis A Antibody Test": "Detects antibodies against hepatitis A virus."
        }
    },
    "Hepatitis B": {
        "tests": ["HBsAg Test", "Liver Function Test"],
        "outcome": "Presence of hepatitis B surface antigen and liver damage.",
        "definition": {
            "HBsAg Test": "Detects hepatitis B surface antigen in the blood.",
            "Liver Function Test": "Measures liver enzymes to check for liver damage."
        }
    },
    
    "Hepatitis C": {
        "tests": ["HCV Antibody Test", "Liver Function Test"],
        "outcome": "Presence of HCV antibodies or liver enzyme abnormalities.",
        "definition": {
            "HCV Antibody Test": "Detects antibodies against the hepatitis C virus.",
            "Liver Function Test": "Measures liver enzyme levels to assess liver damage."
        }
    },
    "Hepatitis D": {
        "tests": ["HDV Antibody Test", "Liver Function Test"],
        "outcome": "Presence of HDV antibodies and elevated liver enzymes.",
        "definition": {
            "HDV Antibody Test": "Detects hepatitis D virus antibodies.",
            "Liver Function Test": "Checks for liver inflammation or damage."
        }
    },
    "Hepatitis E": {
        "tests": ["HEV IgM Antibody Test", "Liver Function Test"],
        "outcome": "Detection of HEV antibodies and liver damage.",
        "definition": {
            "HEV IgM Antibody Test": "Detects hepatitis E virus antibodies.",
            "Liver Function Test": "Monitors liver enzyme levels for damage assessment."
        }
    },
    "Alcoholic hepatitis": {
        "tests": ["Liver Function Test", "Ultrasound", "Alcohol Screening"],
        "outcome": "Elevated liver enzymes and evidence of liver inflammation.",
        "definition": {
            "Liver Function Test": "Measures enzymes to assess liver health.",
            "Ultrasound": "Imaging test to detect liver swelling or damage.",
            "Alcohol Screening": "A test to measure alcohol levels in the blood."
        }
    },
    "Dimorphic hemorrhoids (piles)": {
        "tests": ["Rectal Examination", "Proctoscopy"],
        "outcome": "Swollen veins in the rectum or anal canal.",
        "definition": {
            "Rectal Examination": "Physical examination of the rectum for swollen veins.",
            "Proctoscopy": "A procedure using a small tube to visualize the rectum."
        }
    },
    "Varicose veins": {
        "tests": ["Doppler Ultrasound", "Venography"],
        "outcome": "Poor vein function and enlarged veins.",
        "definition": {
            "Doppler Ultrasound": "Uses sound waves to check blood flow in veins.",
            "Venography": "X-ray with contrast dye to visualize veins."
        }
    },
    "Hypothyroidism": {
        "tests": ["TSH Test", "T3 & T4 Test"],
        "outcome": "Elevated TSH and low thyroid hormones.",
        "definition": {
            "TSH Test": "Measures thyroid-stimulating hormone levels.",
            "T3 & T4 Test": "Checks levels of thyroid hormones in the blood."
        }
    },
    "Hyperthyroidism": {
        "tests": ["TSH Test", "T3 & T4 Test", "Thyroid Scan"],
        "outcome": "Low TSH and high thyroid hormones.",
        "definition": {
            "TSH Test": "Determines thyroid activity by measuring hormone levels.",
            "T3 & T4 Test": "Assess thyroid hormone levels in the blood.",
            "Thyroid Scan": "Imaging test to examine thyroid gland function."
        }
    },
    "Hypoglycemia": {
        "tests": ["Fasting Blood Sugar", "Glucose Tolerance Test"],
        "outcome": "Low blood sugar levels.",
        "definition": {
            "Fasting Blood Sugar": "Measures blood glucose after overnight fasting.",
            "Glucose Tolerance Test": "Evaluates how the body processes sugar."
        }
    },
    "Osteoarthritis": {
        "tests": ["X-ray", "MRI", "Joint Fluid Analysis"],
        "outcome": "Joint space narrowing, cartilage loss, or inflammation.",
        "definition": {
            "X-ray": "Detects bone and joint structure abnormalities.",
            "MRI": "Provides detailed images of cartilage and soft tissues.",
            "Joint Fluid Analysis": "Checks for inflammation or infection in joints."
        }
    },
    "Arthritis": {
        "tests": ["Rheumatoid Factor (RF) Test", "Anti-CCP Test", "ESR Test"],
        "outcome": "Inflammation and presence of autoimmune antibodies.",
        "definition": {
            "Rheumatoid Factor (RF) Test": "Detects antibodies linked to rheumatoid arthritis.",
            "Anti-CCP Test": "Helps diagnose rheumatoid arthritis.",
            "ESR Test": "Measures inflammation in the body."
        }
    },
    "(vertigo) Paroxysmal Positional Vertigo": {
        "tests": ["Dix-Hallpike Test", "Electronystagmography (ENG)"],
        "outcome": "Abnormal eye movements and dizziness triggers.",
        "definition": {
            "Dix-Hallpike Test": "A positioning test to diagnose vertigo.",
            "Electronystagmography (ENG)": "Measures eye movements to detect inner ear problems."
        }
    },
    "Acne": {
        "tests": ["Skin Examination", "Hormonal Panel"],
        "outcome": "Blocked pores, inflammation, or hormonal imbalance.",
        "definition": {
            "Skin Examination": "A visual assessment of skin condition.",
            "Hormonal Panel": "Checks for hormonal imbalances that trigger acne."
        }
    },
    "Urinary tract infection (UTI)": {
        "tests": ["Urinalysis", "Urine Culture"],
        "outcome": "Presence of bacteria and white blood cells in urine.",
        "definition": {
            "Urinalysis": "Examines urine for infection indicators.",
            "Urine Culture": "Identifies bacteria responsible for the infection."
        }
    },
    "Psoriasis": {
        "tests": ["Skin Biopsy", "Dermatological Examination"],
        "outcome": "Abnormal skin cell growth and inflammation.",
        "definition": {
            "Skin Biopsy": "A sample of skin is examined under a microscope.",
            "Dermatological Examination": "Visual assessment of scaly or inflamed skin."
        }
    },
    "Impetigo": {
        "tests": ["Skin Swab Test", "Bacterial Culture"],
        "outcome": "Presence of Staphylococcus or Streptococcus bacteria.",
        "definition": {
            "Skin Swab Test": "A swab is taken from the infected area to identify bacteria.",
            "Bacterial Culture": "Cultivation of bacteria to determine the cause of infection."
        }
    },
    
    "Malaria" : {
        "tests": ["Blood Smear Test", "Rapid Diagnostic Test (RDT)"],
        "outcome": "Presence of Plasmodium parasites in the blood.",
        "definition": {
            "Blood Smear Test": "A microscopic examination of blood samples to detect malaria parasites.",
            "Rapid Diagnostic Test (RDT)": "A test that quickly detects malaria antigens in the blood."
    }
}
}



