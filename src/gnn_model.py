import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import pickle
import pandas as pd
import numpy as np
from config import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import torch.nn as nn
import torch.optim as optim

class GNNModel(nn.Module):
    def __init__(self, num_features, hidden_channels=64):
        super(GNNModel, self).__init__()
        # Node feature processing
        self.node_encoder = nn.Sequential(
            nn.Linear(num_features, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Graph convolution layers
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Edge prediction head
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()  # Ensure output is in [0,1] range
        )
        
    def forward(self, x, edge_index, batch=None):
        # Node feature encoding
        x = self.node_encoder(x)
        
        # Graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Get node embeddings
        node_embeddings = x
        
        # Predict edge weights
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=1)
        edge_pred = self.edge_predictor(edge_features)
        
        return node_embeddings, edge_pred

def normalize_features(x):
    """Normalize node features to [0, 1] range"""
    # Replace NaN values with 0
    x = torch.nan_to_num(x, nan=0.0)
    
    # Get min and max values for each feature
    min_vals = x.min(dim=0)[0]
    max_vals = x.max(dim=0)[0]
    
    # Handle constant features
    max_vals = torch.where(max_vals == min_vals, 
                          torch.ones_like(max_vals), 
                          max_vals)
    
    # Normalize
    x_norm = (x - min_vals) / (max_vals - min_vals)
    
    # Ensure all values are in [0, 1]
    x_norm = torch.clamp(x_norm, 0.0, 1.0)
    
    return x_norm

def normalize_edge_weights(weights):
    """Normalize edge weights to [0, 1] range"""
    # Replace NaN values with 0
    weights = torch.nan_to_num(weights, nan=0.0)
    
    min_weight = weights.min()
    max_weight = weights.max()
    
    if max_weight == min_weight:
        return torch.ones_like(weights)
    
    # Normalize
    weights_norm = (weights - min_weight) / (max_weight - min_weight)
    
    # Ensure all values are in [0, 1]
    weights_norm = torch.clamp(weights_norm, 0.0, 1.0)
    
    return weights_norm

def prepare_graph_data():
    """Prepare graph data for GNN training"""
    print("Preparing graph data...")
    
    # Load the knowledge graph
    with open(os.path.join(DATA_DIR, 'knowledge_graph.pkl'), 'rb') as f:
        G = pickle.load(f)
    
    # Load node features
    node_features = pd.read_csv(os.path.join(DATA_DIR, 'node_features.csv'))
    
    # Get the number of features from the first node
    first_node = list(G.nodes())[0]
    num_features = len(node_features[first_node].values)
    
    # Create node feature matrix
    num_nodes = len(G.nodes())
    x = torch.zeros((num_nodes, num_features))
    
    # Map node names to indices
    node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
    print(f"node_to_idx: {node_to_idx}")
    print(f"G.nodes(): {list(G.nodes())}")
    print(f"node_features columns: {list(node_features.columns)}")
    
    # Fill feature matrix
    for node, idx in node_to_idx.items():
        features = node_features[node].values
        if len(features) != num_features:
            print(f"Warning: Node {node} has {len(features)} features, expected {num_features}")
            # Pad or truncate features to match expected size
            if len(features) < num_features:
                features = np.pad(features, (0, num_features - len(features)))
            else:
                features = features[:num_features]
        x[idx] = torch.tensor(features, dtype=torch.float)
    
    # Check for NaN values before normalization
    if torch.isnan(x).any():
        print("Warning: NaN values found in node features before normalization")
        print(f"Number of NaN values: {torch.isnan(x).sum().item()}")
    
    # Normalize node features
    x = normalize_features(x)
    
    # Create edge index tensor and edge names
    edge_index = []
    edge_names = []
    for u, v in G.edges():
        edge_index.append([node_to_idx[u], node_to_idx[v]])
        edge_names.append(f"{u}→{v}")
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    print(f"x.shape: {x.shape}")
    print(f"edge_index max: {edge_index.max().item()}, min: {edge_index.min().item()}")
    print(f"edge_index: {edge_index}")
    
    # Create and normalize edge weights
    edge_weights = torch.tensor([G[u][v]['weight'] for u, v in G.edges()], 
                              dtype=torch.float)
    
    # Check for NaN values in edge weights
    if torch.isnan(edge_weights).any():
        print("Warning: NaN values found in edge weights before normalization")
        print(f"Number of NaN values: {torch.isnan(edge_weights).sum().item()}")
    
    edge_weights = normalize_edge_weights(edge_weights)
    
    # Verify normalization
    print(f"Graph data prepared:")
    print(f"- Number of nodes: {num_nodes}")
    print(f"- Number of features per node: {num_features}")
    print(f"- Number of edges: {len(edge_index[0])}")
    print(f"- Edge weight range: [{edge_weights.min().item():.4f}, {edge_weights.max().item():.4f}]")
    print(f"- Node feature range: [{x.min().item():.4f}, {x.max().item():.4f}]")
    
    # Final verification
    if torch.isnan(x).any() or torch.isnan(edge_weights).any():
        raise ValueError("NaN values found after normalization")
    
    if not (0 <= x.min() <= x.max() <= 1):
        raise ValueError("Node features not properly normalized to [0, 1] range")
    
    if not (0 <= edge_weights.min() <= edge_weights.max() <= 1):
        raise ValueError("Edge weights not properly normalized to [0, 1] range")
    
    return x, edge_index, edge_weights, edge_names

def train_gnn(model, x, edge_index, edge_weights, epochs=100, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.BCELoss()  # Binary Cross Entropy for [0,1] predictions
    
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    # Ensure edge_weights shape matches edge_pred ([N, 1])
    edge_weights = edge_weights.unsqueeze(1)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        _, edge_pred = model(x, edge_index)
        
        # Calculate loss
        loss = criterion(edge_pred, edge_weights)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Learning rate scheduling
        scheduler.step(loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}')
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
    
    return model

def evaluate_gnn(model, x, edge_index, edge_weights, edge_names, save_dir='data/evaluation'):
    """Evaluate the model and return metrics"""
    model.eval()
    with torch.no_grad():
        # Get predictions
        node_embeddings, edge_predictions = model(x, edge_index)
        
        # Convert to numpy for evaluation
        y_true = edge_weights.numpy()
        y_pred = edge_predictions.numpy()
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        print(f"\nMean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        
        # Print example predictions
        print("\nExample predictions:")
        for i in range(min(5, len(y_pred))):
            print(f"Edge {i}: Predicted = {float(y_pred[i]):.4f}, Actual = {float(y_true[i]):.4f}")
        
        # Create evaluation directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save node embeddings for later analysis
        np.save(os.path.join(save_dir, 'node_embeddings.npy'), node_embeddings.numpy())
        
        return mse, mae, node_embeddings

def main():
    """Main function to run GNN training and evaluation"""
    print("Starting GNN training pipeline...")
    
    try:
        # Prepare data
        x, edge_index, edge_weights, edge_names = prepare_graph_data()
        
        # Initialize model
        model = GNNModel(num_features=x.size(1))
        
        # Train model
        model = train_gnn(model, x, edge_index, edge_weights)
        
        # Evaluate model
        predictions = evaluate_gnn(model, x, edge_index, edge_weights, edge_names)
        
        # Save model
        torch.save(model.state_dict(), os.path.join(DATA_DIR, 'gnn_model.pt'))
        
        print("GNN training complete!")
        
    except Exception as e:
        print(f"Error in GNN training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 