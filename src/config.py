# src/config.py
import os

# Data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
NHANES_DATA_DIR = os.path.join(DATA_DIR, 'nhanes')
MIMIC_DATA_DIR = os.path.join(DATA_DIR, 'mimic')

# Create directories if they don't exist
os.makedirs(NHANES_DATA_DIR, exist_ok=True)
os.makedirs(MIMIC_DATA_DIR, exist_ok=True)

# NHANES data years to download
NHANES_YEARS = ['2017-2018', '2019-2020']

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