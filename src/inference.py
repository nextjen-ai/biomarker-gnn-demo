import torch
import numpy as np
import pandas as pd
from gnn_model import GNNModel, normalize_features
from config import DATA_DIR
import os

def load_model(model_path):
    """Load a trained GNN model"""
    # Load the model architecture
    model = GNNModel(num_features=6)  # 6 features per node as seen in training
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model

def prepare_patient_data(patient_data):
    """
    Prepare patient data for inference
    
    Parameters:
    patient_data: dict with biomarker values
        Required keys: 'trestbps', 'chol', 'thalach', 'oldpeak'
    """
    # Define the biomarkers we're working with
    biomarkers = ['trestbps', 'chol', 'thalach', 'oldpeak']
    
    # Create a feature matrix for the patient
    # Each biomarker gets 6 features (as in training)
    x = torch.zeros((8, 6))  # 8 nodes (4 biomarkers + 4 diseases)
    
    # Fill in biomarker features
    for i, biomarker in enumerate(biomarkers):
        value = patient_data[biomarker]
        # Create 6 features for each biomarker
        features = [
            value,  # raw value
            value/200,  # normalized value
            value/100,  # another normalized value
            value/50,   # another normalized value
            value/25,   # another normalized value
            value/10    # another normalized value
        ]
        x[i] = torch.tensor(features, dtype=torch.float)
    
    # Normalize features
    x = normalize_features(x)
    
    # Create edge index (same as in training)
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2, 3],
        [4, 6, 4, 7, 4, 5, 4]
    ], dtype=torch.long)
    
    return x, edge_index

def predict_disease_risks(model, x, edge_index):
    """Make predictions for disease risks"""
    with torch.no_grad():
        _, edge_predictions = model(x, edge_index)
        predictions = edge_predictions.numpy()
    
    # Map predictions to diseases
    disease_predictions = {
        'heart_disease': float(predictions[0]),  # trestbps -> heart_disease
        'hypertension': float(predictions[1]),   # trestbps -> hypertension
        'high_cholesterol': float(predictions[3]),  # chol -> high_cholesterol
        'angina': float(predictions[5])          # thalach -> angina
    }
    
    return disease_predictions

def main():
    """Main function to run inference on patient data"""
    # Example patient data
    patient_data = {
        'trestbps': 145,  # resting blood pressure
        'chol': 250,      # serum cholesterol
        'thalach': 150,   # maximum heart rate
        'oldpeak': 1.5    # ST depression
    }
    
    try:
        # Load the trained model
        model_path = os.path.join(DATA_DIR, 'gnn_model.pt')
        model = load_model(model_path)
        
        # Prepare patient data
        x, edge_index = prepare_patient_data(patient_data)
        
        # Make predictions
        predictions = predict_disease_risks(model, x, edge_index)
        
        # Print results
        print("\nDisease Risk Predictions:")
        print("-" * 30)
        for disease, risk in predictions.items():
            print(f"{disease}: {risk:.2%}")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    main() 