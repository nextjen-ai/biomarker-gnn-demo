import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import networkx as nx
import os
import pickle
import pandas as pd

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

def load_knowledge_graph(data_dir):
    """Load the knowledge graph and node features"""
    # Load graph structure
    with open(os.path.join(data_dir, 'knowledge_graph.pkl'), 'rb') as f:
        G = pickle.load(f)
    
    # Load node features
    node_features = pd.read_csv(os.path.join(data_dir, 'node_features.csv'), index_col=0)
    
    return G, node_features

def prepare_graph_data(patient_data, heart_disease_present, G, node_features):
    """
    Prepare graph data for a single patient using the knowledge graph structure
    
    Args:
        patient_data (dict): Dictionary containing biomarker values
        heart_disease_present (bool): Whether the patient has heart disease
        G (nx.DiGraph): Knowledge graph
        node_features (pd.DataFrame): Node features from knowledge graph
        
    Returns:
        tuple: (node_features, edge_index, edge_weight)
    """
    # Create node features using knowledge graph features
    node_feature_list = []
    for biomarker in G.nodes():
        if G.nodes[biomarker]['type'] == 'biomarker':
            # Get the biomarker value
            value = patient_data[biomarker]
            
            # Normalize using knowledge graph statistics
            stats = node_features[biomarker]
            normalized = (value - stats['min']) / (stats['max'] - stats['min'])
            
            # Add statistical features
            node_feature_list.extend([
                normalized,
                stats['mean'],
                stats['std']
            ])
    
    # Add heart disease node features
    disease_stats = node_features['heart_disease']
    node_feature_list.extend([
        float(heart_disease_present),
        disease_stats['prevalence'],
        disease_stats['severity']
    ])
    
    # Convert to tensor
    node_features = torch.tensor(node_feature_list, dtype=torch.float).view(-1, 3)
    
    # Create edge index and weights from knowledge graph
    edge_index = []
    edge_weight = []
    
    for u, v, data in G.edges(data=True):
        # Add edge in both directions
        edge_index.append([list(G.nodes()).index(u), list(G.nodes()).index(v)])
        edge_index.append([list(G.nodes()).index(v), list(G.nodes()).index(u)])
        
        # Use correlation as edge weight
        weight = data['weight']
        edge_weight.extend([weight, weight])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    
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