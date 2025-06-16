import torch
import numpy as np
import pandas as pd
from src.gnn_model import GNNModel, normalize_features
from config import DATA_DIR
import os

def load_model(model_path='src/model.pth'):
    """Load the trained model"""
    model = GNNModel(num_node_features=1)  # 1 feature per node (normalized biomarker value)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def prepare_patient_data(patient_data):
    # patient_data is a dict of {biomarker: value}
    node_features = []
    for biomarker in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']:
        value = patient_data[biomarker]
        normalized = normalize_features(biomarker, value)
        node_features.append(normalized)
    # Add a dummy heart disease node (value doesn't matter for inference)
    node_features.append(0.0)
    x = torch.tensor(node_features, dtype=torch.float).unsqueeze(1)
    # Build edge_index as in training
    num_biomarkers = 13
    edge_index = []
    for i in range(num_biomarkers):
        edge_index.append([i, num_biomarkers])
        edge_index.append([num_biomarkers, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    return x, edge_index

def predict_disease_risks(model, x, edge_index):
    model.eval()
    with torch.no_grad():
        # Create edge_weight and batch tensors for inference
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
        batch = torch.zeros(x.size(0), dtype=torch.long)
        # Forward pass
        predictions = model(x, edge_index, edge_weight, batch)
    return predictions

def main():
    model_path = 'data/model.pth'
    model = load_model(model_path)
    
    # Example patient data
    patient_data = {
        'age': 50,       # age in years
        'sex': 0,        # 1 = male, 0 = female
        'cp': 2,         # chest pain type (1, 2, 3, 4)
        'trestbps': 145,  # resting blood pressure
        'chol': 250,      # serum cholesterol
        'thalach': 150,   # maximum heart rate
        'oldpeak': 1.5,   # ST depression
        'fbs': 1,         # fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
        'restecg': 0,     # resting electrocardiographic results (0, 1, 2)
        'exang': 0,       # exercise induced angina (1 = yes, 0 = no)
        'slope': 2,       # slope of peak exercise ST segment (1, 2, or 3)
        'ca': 2,          # number of major vessels (0-3)
        'thal': 2         # thalassemia (1, 2, 3)
    }
    
    # Prepare patient data for inference
    x, edge_index = prepare_patient_data(patient_data)
    
    # Get predictions
    predictions = predict_disease_risks(model, x, edge_index)
    
    # Extract the prediction value
    risk = predictions.item()  # Assuming binary classification
    
    # Display the prediction
    print("\nDisease Risk Predictions:")
    print("------------------------------")
    print(f"Heart Disease Risk: {risk:.4f}")

if __name__ == "__main__":
    main() 