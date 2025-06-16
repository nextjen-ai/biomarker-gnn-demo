import torch
from torch_geometric.data import Data, DataLoader
from src.gnn_model import GNNModel, prepare_graph_data, train_model, evaluate_model
import numpy as np

def create_sample_data(num_samples=100):
    """Create sample patient data for testing"""
    data_list = []
    
    for _ in range(num_samples):
        # Generate random patient data
        patient_data = {
            'age': np.random.randint(30, 80),
            'sex': np.random.randint(0, 2),
            'cp': np.random.randint(0, 4),
            'trestbps': np.random.randint(90, 200),
            'chol': np.random.randint(100, 600),
            'fbs': np.random.randint(0, 2),
            'restecg': np.random.randint(0, 3),
            'thalach': np.random.randint(60, 250),
            'exang': np.random.randint(0, 2),
            'oldpeak': np.random.uniform(0, 6),
            'slope': np.random.randint(0, 3),
            'ca': np.random.randint(0, 4),
            'thal': np.random.randint(0, 4)
        }
        
        # Randomly assign heart disease (1) or no heart disease (0)
        heart_disease = np.random.randint(0, 2)
        
        # Prepare graph data
        x, edge_index, edge_weight = prepare_graph_data(patient_data, heart_disease)
        
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
    
    # Create sample data
    print("Creating sample data...")
    data_list = create_sample_data(num_samples=100)
    
    # Split into train and test sets
    train_size = int(0.8 * len(data_list))
    train_data = data_list[:train_size]
    test_data = data_list[train_size:]
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)
    
    # Initialize model
    print("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(num_node_features=1).to(device)  # 13 biomarkers
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()
    
    # Training loop
    print("Starting training...")
    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate_model(model, test_loader, device)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Test Accuracy: {test_acc:.4f}')
    
    print("\nTraining completed!")
    
    # Save the trained model
    print("Saving model...")
    torch.save(model.state_dict(), 'data/model.pth')
    print("Model saved to data/model.pth")

if __name__ == "__main__":
    main() 