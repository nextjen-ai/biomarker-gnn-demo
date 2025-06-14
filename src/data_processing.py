# src/data_processing.py
import pandas as pd
import numpy as np
import requests
import os
from tqdm import tqdm
from config import *

def download_heart_disease_data():
    """Download Heart Disease dataset from UCI"""
    print("Downloading Heart Disease data...")
    
    # UCI Heart Disease dataset URL
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    output_file = os.path.join(DATA_DIR, 'heart_disease.csv')
    
    if not os.path.exists(output_file):
        print(f"Downloading data from {data_url}...")
        try:
            # Download the file
            response = requests.get(data_url)
            response.raise_for_status()
            
            # Save the data
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            print("Successfully downloaded Heart Disease data")
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading data: {str(e)}")
            raise

def process_heart_disease_data():
    """Process Heart Disease data into a clean format"""
    print("Processing Heart Disease data...")
    
    try:
        # Read the data
        data_file = os.path.join(DATA_DIR, 'heart_disease.csv')
        
        # Define column names
        columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        
        # Read CSV with column names
        data = pd.read_csv(data_file, names=columns)
        
        # Clean data
        data = data.replace('?', np.nan)
        data = data.astype(float)
        
        # Save processed data
        output_file = os.path.join(DATA_DIR, 'processed_heart_disease.csv')
        data.to_csv(output_file, index=False)
        
        return data
        
    except Exception as e:
        print(f"Error processing Heart Disease data: {str(e)}")
        raise

def create_initial_knowledge_graph(processed_data):
    """Create initial knowledge graph structure"""
    print("Creating knowledge graph structure...")
    
    # Define biomarkers based on the dataset
    biomarkers = [
        'trestbps',  # resting blood pressure
        'chol',      # serum cholesterol
        'thalach',   # maximum heart rate
        'oldpeak'    # ST depression
    ]
    
    # Define diseases/conditions
    diseases = [
        'heart_disease',
        'angina',
        'hypertension',
        'high_cholesterol'
    ]
    
    # Create nodes for biomarkers
    biomarker_nodes = pd.DataFrame({
        'id': biomarkers,
        'type': 'biomarker',
        'name': biomarkers
    })
    
    # Create nodes for diseases
    disease_nodes = pd.DataFrame({
        'id': diseases,
        'type': 'disease',
        'name': diseases
    })
    
    # Save nodes
    biomarker_nodes.to_csv(os.path.join(DATA_DIR, 'biomarker_nodes.csv'), index=False)
    disease_nodes.to_csv(os.path.join(DATA_DIR, 'disease_nodes.csv'), index=False)
    
    # Create relationships based on correlations
    relationships = []
    
    # Calculate correlations between biomarkers and target
    correlations = processed_data[biomarkers + ['target']].corr()['target']
    
    for biomarker in biomarkers:
        for disease in diseases:
            # Use correlation as confidence score
            confidence = abs(correlations[biomarker])
            relationships.append({
                'source': biomarker,
                'target': disease,
                'type': 'associated_with',
                'confidence': confidence
            })
    
    # Save relationships
    pd.DataFrame(relationships).to_csv(
        os.path.join(DATA_DIR, 'relationships.csv'), 
        index=False
    )

def main():
    """Main function to run data processing pipeline"""
    print("Starting data processing pipeline...")
    
    try:
        # Download data
        download_heart_disease_data()
        
        # Process data
        processed_data = process_heart_disease_data()
        
        # Create knowledge graph structure
        create_initial_knowledge_graph(processed_data)
        
        print("Data processing complete!")
        
    except Exception as e:
        print(f"Error in data processing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()