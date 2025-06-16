import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
import os
from config import DATA_DIR, MODEL_DIR, BIOMARKERS
from src.gnn_model import (
    GNNModel, load_knowledge_graph, prepare_graph_data,
    train_model, evaluate_model
)
from sklearn.model_selection import KFold

def create_old_model_data(num_samples=100):
    """Create data using the old approach (without knowledge graph)"""
    data_list = []
    
    # Define realistic ranges for each biomarker
    ranges = {
        'trestbps': (90, 200),    # resting blood pressure
        'chol': (100, 600),       # serum cholesterol
        'thalach': (60, 250),     # maximum heart rate
        'oldpeak': (0, 6),        # ST depression
        'fbs': (0, 1),            # fasting blood sugar
        'exang': (0, 1),          # exercise induced angina
        'slope': (0, 2),          # slope of peak exercise ST segment
        'ca': (0, 3)              # number of major vessels
    }
    
    # Define risk factors (higher values increase heart disease risk)
    risk_factors = {
        'trestbps': lambda x: (x - ranges['trestbps'][0]) / (ranges['trestbps'][1] - ranges['trestbps'][0]),
        'chol': lambda x: (x - ranges['chol'][0]) / (ranges['chol'][1] - ranges['chol'][0]),
        'thalach': lambda x: 1 - (x - ranges['thalach'][0]) / (ranges['thalach'][1] - ranges['thalach'][0]),  # inverse
        'oldpeak': lambda x: (x - ranges['oldpeak'][0]) / (ranges['oldpeak'][1] - ranges['oldpeak'][0]),
        'fbs': lambda x: x,
        'exang': lambda x: x,
        'slope': lambda x: x / ranges['slope'][1],
        'ca': lambda x: x / ranges['ca'][1]
    }
    
    for _ in range(num_samples):
        # Generate random patient data
        patient_data = {}
        total_risk = 0
        
        for biomarker, (min_val, max_val) in ranges.items():
            if biomarker in ['fbs', 'exang']:
                value = np.random.randint(min_val, max_val + 1)
            else:
                value = np.random.uniform(min_val, max_val)
            patient_data[biomarker] = value
            total_risk += risk_factors[biomarker](value)
        
        # Normalize total risk to [0,1]
        total_risk /= len(ranges)
        
        # Assign heart disease based on risk (with some randomness)
        heart_disease = 1 if total_risk + np.random.normal(0, 0.1) > 0.5 else 0
        
        # Create node features
        node_features = []
        for biomarker in BIOMARKERS:
            value = patient_data[biomarker]
            # Normalize using ranges
            min_val, max_val = ranges[biomarker]
            normalized = (value - min_val) / (max_val - min_val)
            node_features.append(normalized)
        
        # Add heart disease node
        node_features.append(float(heart_disease))
        
        # Convert to tensor
        x = torch.tensor(node_features, dtype=torch.float).unsqueeze(1)
        
        # Create edge index
        num_biomarkers = len(BIOMARKERS)
        edge_index = []
        for i in range(num_biomarkers):
            edge_index.append([i, num_biomarkers])
            edge_index.append([num_biomarkers, i])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Create edge weights based on risk factors
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
        for i, biomarker in enumerate(BIOMARKERS):
            risk = risk_factors[biomarker](patient_data[biomarker])
            edge_weight[i*2] = risk
            edge_weight[i*2+1] = risk
        
        # Create data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=torch.tensor([heart_disease], dtype=torch.float)
        )
        
        data_list.append(data)
    
    return data_list

def train_and_evaluate_model(model, train_loader, test_loader, device):
    """Train and evaluate a model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()
    
    best_acc = 0
    for epoch in range(50):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate_model(model, test_loader, device)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/50:')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Test Accuracy: {test_acc:.4f}')
    
    return best_acc

def cross_validate_old_model():
    """Cross-validate the old model version"""
    print("\nCross-validating old model version...")
    
    # Create data
    data_list = create_old_model_data(num_samples=1000)
    
    # Setup cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(data_list)):
        print(f"\nFold {fold + 1}/5")
        
        # Split data
        train_data = [data_list[i] for i in train_idx]
        test_data = [data_list[i] for i in test_idx]
        
        # Create loaders
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32)
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GNNModel(num_node_features=1).to(device)
        
        # Train and evaluate
        acc = train_and_evaluate_model(model, train_loader, test_loader, device)
        accuracies.append(acc)
    
    return np.mean(accuracies), np.std(accuracies)

def cross_validate_new_model():
    """Cross-validate the new model version with knowledge graph"""
    print("\nCross-validating new model version with knowledge graph...")
    
    # Load knowledge graph
    G, node_features = load_knowledge_graph(DATA_DIR)
    
    # Create data
    data_list = []
    for _ in range(1000):
        # Generate random patient data using knowledge graph statistics
        patient_data = {}
        for biomarker in BIOMARKERS:
            stats = node_features[biomarker]
            value = np.random.uniform(stats['min'], stats['max'])
            patient_data[biomarker] = value
        
        # Randomly assign heart disease
        heart_disease = np.random.randint(0, 2)
        
        # Prepare graph data
        x, edge_index, edge_weight = prepare_graph_data(patient_data, heart_disease, G, node_features)
        
        # Create data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=torch.tensor([heart_disease], dtype=torch.float)
        )
        
        data_list.append(data)
    
    # Setup cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(data_list)):
        print(f"\nFold {fold + 1}/5")
        
        # Split data
        train_data = [data_list[i] for i in train_idx]
        test_data = [data_list[i] for i in test_idx]
        
        # Create loaders
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32)
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GNNModel(num_node_features=3).to(device)
        
        # Train and evaluate
        acc = train_and_evaluate_model(model, train_loader, test_loader, device)
        accuracies.append(acc)
    
    return np.mean(accuracies), np.std(accuracies)

def main():
    print("Starting model comparison with cross-validation...")
    
    # Cross-validate old model
    old_mean, old_std = cross_validate_old_model()
    
    # Cross-validate new model
    new_mean, new_std = cross_validate_new_model()
    
    # Print comparison
    print("\nModel Comparison (5-fold cross-validation):")
    print("-------------------------------------------")
    print(f"Old Model (without knowledge graph): {old_mean:.2%} ± {old_std:.2%}")
    print(f"New Model (with knowledge graph): {new_mean:.2%} ± {new_std:.2%}")
    print(f"Improvement: {(new_mean - old_mean):.2%}")

if __name__ == "__main__":
    main() 