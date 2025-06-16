import torch
from src.gnn_model import GNNModel, load_knowledge_graph, prepare_graph_data
from config import DATA_DIR, MODEL_DIR, BIOMARKERS
import os

def load_model(model_path):
    """Load the trained model"""
    model = GNNModel(num_node_features=3)  # 3 features per node
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_heart_disease(patient_data, model, G, node_features):
    """
    Predict heart disease probability for a patient
    
    Args:
        patient_data (dict): Dictionary containing biomarker values
        model (GNNModel): Trained model
        G (nx.DiGraph): Knowledge graph
        node_features (pd.DataFrame): Node features from knowledge graph
        
    Returns:
        float: Probability of heart disease
    """
    # Prepare graph data
    x, edge_index, edge_weight = prepare_graph_data(patient_data, False, G, node_features)
    
    # Add batch dimension only to node features
    x = x.unsqueeze(0)  # [num_nodes, num_features] -> [1, num_nodes, num_features]
    batch = torch.zeros(x.size(0), dtype=torch.long)
    
    # Make prediction
    with torch.no_grad():
        out = model(x, edge_index, edge_weight, batch)
        probability = out.item()
    
    return probability

def main():
    # Load knowledge graph
    print("Loading knowledge graph...")
    G, node_features = load_knowledge_graph(DATA_DIR)
    
    # Load model
    print("Loading model...")
    model_path = os.path.join(MODEL_DIR, 'model.pth')
    model = load_model(model_path)
    
    # Example patient data
    patient_data = {
        'trestbps': 145,  # resting blood pressure
        'chol': 233,      # serum cholesterol
        'thalach': 150,   # maximum heart rate
        'oldpeak': 2.3,   # ST depression
        'fbs': 1,         # fasting blood sugar > 120 mg/dl
        'exang': 0,       # exercise induced angina
        'slope': 0,       # slope of peak exercise ST segment
        'ca': 0           # number of major vessels colored by flourosopy
    }
    
    # Make prediction
    print("\nMaking prediction for patient...")
    probability = predict_heart_disease(patient_data, model, G, node_features)
    
    print(f"\nHeart Disease Probability: {probability:.2%}")
    print("\nBiomarker Values:")
    for biomarker, value in patient_data.items():
        print(f"{biomarker}: {value}")

if __name__ == "__main__":
    main() 