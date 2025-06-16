import torch
from torch_geometric.data import Data, DataLoader
from src.gnn_model import GNNModel, prepare_graph_data, train_model, evaluate_model, load_knowledge_graph
import numpy as np
import os
from config import (
    DATA_DIR, MODEL_DIR, BIOMARKERS, DISEASES,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, HIDDEN_DIM
)

def create_sample_data(num_samples=100, G=None, node_features=None):
    """Create sample patient data for testing using knowledge graph structure"""
    data_list = []
    
    # Get biomarker names from knowledge graph
    biomarkers = [n for n, d in G.nodes(data=True) if d['type'] == 'biomarker']
    
    for _ in range(num_samples):
        # Generate random patient data using knowledge graph statistics
        patient_data = {}
        for biomarker in biomarkers:
            stats = node_features[biomarker]
            # Generate value within min-max range
            value = np.random.uniform(stats['min'], stats['max'])
            patient_data[biomarker] = value
        
        # Randomly assign heart disease (1) or no heart disease (0)
        heart_disease = np.random.randint(0, 2)
        
        # Prepare graph data using knowledge graph
        x, edge_index, edge_weight = prepare_graph_data(patient_data, heart_disease, G, node_features)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=torch.tensor([heart_disease], dtype=torch.float)
        )
        
        data_list.append(data)
    
    return data_list

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load knowledge graph
    print("Loading knowledge graph...")
    G, node_features = load_knowledge_graph(DATA_DIR)
    
    # Create sample data
    print("Creating sample data...")
    data_list = create_sample_data(num_samples=100, G=G, node_features=node_features)
    
    # Split into train and test sets
    train_size = int(0.8 * len(data_list))
    train_data = data_list[:train_size]
    test_data = data_list[train_size:]
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    
    # Initialize model
    print("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Each node has 3 features: normalized value, mean, std
    model = GNNModel(num_node_features=3).to(device)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()
    
    # Training loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate_model(model, test_loader, device)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{NUM_EPOCHS}:')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Test Accuracy: {test_acc:.4f}')
    
    print("\nTraining completed!")
    
    # Save the trained model
    print("Saving model...")
    model_path = os.path.join(MODEL_DIR, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main() 