# src/config.py
import os

# Data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
MODEL_DIR = os.path.join(DATA_DIR, 'models')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Biomarkers to focus on (from heart disease dataset)
BIOMARKERS = [
    'trestbps',  # resting blood pressure
    'chol',      # serum cholesterol
    'thalach',   # maximum heart rate
    'oldpeak',   # ST depression
    'fbs',       # fasting blood sugar > 120 mg/dl
    'exang',     # exercise induced angina
    'slope',     # slope of peak exercise ST segment
    'ca'         # number of major vessels colored by flourosopy
]

# Diseases to focus on
DISEASES = [
    'heart_disease'
]

# Model parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.01
NUM_EPOCHS = 50
HIDDEN_DIM = 64