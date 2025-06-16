import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np

class GNNModel(nn.Module):
    def __init__(self, num_node_features, num_classes=1):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index, edge_weight, batch):
        # First Graph Convolution Layer
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second Graph Convolution Layer
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        
        # Global Pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return x

def prepare_graph_data(patient_data, heart_disease_present):
    """
    Prepare graph data for a single patient
    
    Args:
        patient_data (dict): Dictionary containing biomarker values
        heart_disease_present (bool): Whether the patient has heart disease
        
    Returns:
        tuple: (node_features, edge_index, edge_weight)
    """
    # Create node features (just the normalized biomarker values)
    node_features = []
    for biomarker in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']:
        value = patient_data[biomarker]
        # Normalize to [0,1] range based on typical ranges for each biomarker
        if biomarker == 'age':
            normalized = value / 100.0  # Assuming max age of 100
        elif biomarker == 'sex':
            normalized = value  # Already binary
        elif biomarker == 'cp':
            normalized = value / 3.0  # 0-3 range
        elif biomarker == 'trestbps':
            normalized = value / 200.0  # Typical range 90-200
        elif biomarker == 'chol':
            normalized = value / 600.0  # Typical range 100-600
        elif biomarker == 'fbs':
            normalized = value  # Already binary
        elif biomarker == 'restecg':
            normalized = value / 2.0  # 0-2 range
        elif biomarker == 'thalach':
            normalized = value / 250.0  # Typical range 60-250
        elif biomarker == 'exang':
            normalized = value  # Already binary
        elif biomarker == 'oldpeak':
            normalized = value / 6.0  # Typical range 0-6
        elif biomarker == 'slope':
            normalized = value / 2.0  # 0-2 range
        elif biomarker == 'ca':
            normalized = value / 3.0  # 0-3 range
        elif biomarker == 'thal':
            normalized = value / 3.0  # 0-3 range
            
        node_features.append(normalized)
    
    # Add heart disease node feature (0 or 1)
    node_features.append(float(heart_disease_present))
    
    # Convert to tensor of shape [num_nodes, 1]
    node_features = torch.tensor(node_features, dtype=torch.float).unsqueeze(1)
    
    # Create edge index (connecting each biomarker to heart disease)
    num_biomarkers = 13  # Number of biomarkers (excluding heart disease node)
    edge_index = []
    for i in range(num_biomarkers):
        edge_index.append([i, num_biomarkers])  # Connect biomarker to heart disease
        edge_index.append([num_biomarkers, i])  # Connect heart disease to biomarker
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # Create edge weights based on heart disease presence
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    if heart_disease_present:
        edge_weight *= 1.0  # Stronger connections when heart disease is present
    else:
        edge_weight *= 0.5  # Weaker connections when heart disease is absent
    
    return node_features, edge_index, edge_weight

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
        loss = criterion(out, batch.y.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        
    return total_loss / len(train_loader.dataset)

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
            pred = (out > 0.5).float()
            correct += (pred == batch.y.unsqueeze(1)).sum().item()
            total += batch.num_graphs
            
    return correct / total

def normalize_features(biomarker, value):
    """
    Normalize a biomarker value to [0,1] range based on typical clinical ranges
    
    Args:
        biomarker (str): Name of the biomarker
        value (float): Raw biomarker value
        
    Returns:
        float: Normalized value in [0,1] range
    """
    if biomarker == 'age':
        return value / 100.0  # Assuming max age of 100
    elif biomarker == 'sex':
        return value  # Already binary
    elif biomarker == 'cp':
        return value / 3.0  # 0-3 range
    elif biomarker == 'trestbps':
        return value / 200.0  # Typical range 90-200
    elif biomarker == 'chol':
        return value / 600.0  # Typical range 100-600
    elif biomarker == 'fbs':
        return value  # Already binary
    elif biomarker == 'restecg':
        return value / 2.0  # 0-2 range
    elif biomarker == 'thalach':
        return value / 250.0  # Typical range 60-250
    elif biomarker == 'exang':
        return value  # Already binary
    elif biomarker == 'oldpeak':
        return value / 6.0  # Typical range 0-6
    elif biomarker == 'slope':
        return value / 2.0  # 0-2 range
    elif biomarker == 'ca':
        return value / 3.0  # 0-3 range
    elif biomarker == 'thal':
        return value / 3.0  # 0-3 range
    else:
        raise ValueError(f"Unknown biomarker: {biomarker}") 