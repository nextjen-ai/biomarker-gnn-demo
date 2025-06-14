# src/config.py
import os

# Data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# Biomarkers to focus on
BIOMARKERS = [
    'GLU',  # Glucose
    'CHOL', # Cholesterol
    'HDL',  # HDL Cholesterol
    'LDL',  # LDL Cholesterol
    'TRIG', # Triglycerides
    'HBA1C', # Hemoglobin A1c
    'CRP',  # C-reactive protein
    'ALB',  # Albumin
    'CRE',  # Creatinine
    'ALT'   # Alanine aminotransferase
]

# Diseases to focus on
DISEASES = [
    'diabetes',
    'hypertension',
    'cardiovascular_disease',
    'kidney_disease',
    'liver_disease'
]