import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import networkx as nx
from config import *

def plot_training_progress(losses, save_path=None):
    """Plot training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, threshold=0.5, save_path=None):
    """Plot confusion matrix for edge predictions"""
    # Convert predictions to binary using threshold
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_true_binary = (y_true >= threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_true, y_pred, threshold=0.5, save_path=None):
    """Plot ROC curve for edge predictions"""
    # Convert continuous predictions to binary labels
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_node_embeddings(model, x, edge_index, node_names, save_path=None):
    """Plot node embeddings using t-SNE"""
    from sklearn.manifold import TSNE
    
    # Get node embeddings
    model.eval()
    with torch.no_grad():
        # Get embeddings from the last GCN layer
        x = model.conv1(x, edge_index)
        x = torch.relu(x)
        x = model.conv2(x, edge_index)
        x = torch.relu(x)
    
    # Convert to numpy
    embeddings = x.numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    # Add node labels
    for i, name in enumerate(node_names):
        plt.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    plt.title('Node Embeddings (t-SNE)')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_detailed_predictions(y_true, y_pred, edge_names, save_path=None):
    """Create a detailed analysis of predictions"""
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Prediction vs Actual scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.6)
    ax1.plot([0, 1], [0, 1], 'r--')
    ax1.set_xlabel('Actual Edge Weight')
    ax1.set_ylabel('Predicted Edge Weight')
    ax1.set_title('Predicted vs Actual Edge Weights')
    
    # 2. Error distribution
    errors = y_pred - y_true
    ax2.hist(errors, bins=20, alpha=0.6)
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Count')
    ax2.set_title('Error Distribution')
    
    # 3. Performance by weight range
    weight_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    mae_by_range = []
    for low, high in weight_ranges:
        mask = (y_true >= low) & (y_true < high)
        if mask.any():
            mae = np.mean(np.abs(y_pred[mask] - y_true[mask]))
            mae_by_range.append(mae)
        else:
            mae_by_range.append(0)
    
    ax3.bar(range(len(weight_ranges)), mae_by_range)
    ax3.set_xticks(range(len(weight_ranges)))
    ax3.set_xticklabels([f'{low:.1f}-{high:.1f}' for low, high in weight_ranges])
    ax3.set_xlabel('Actual Weight Range')
    ax3.set_ylabel('Mean Absolute Error')
    ax3.set_title('Error by Weight Range')
    
    # 4. Top predictions table
    # Sort by absolute error
    errors = np.abs(y_pred - y_true)
    sorted_indices = np.argsort(errors)
    
    # Create table data
    table_data = []
    for i in range(min(5, len(y_pred))):
        idx = int(sorted_indices[i][0])  # Extract a single element and convert to Python integer
        table_data.append([
            edge_names[idx],
            f'{y_true[idx]:.4f}',
            f'{y_pred[idx][0]:.4f}',  # Extract a single element before formatting
            f'{errors[idx]:.4f}'
        ])
    
    # Create table
    ax4.axis('tight')
    ax4.axis('off')
    table = ax4.table(
        cellText=table_data,
        colLabels=['Edge', 'Actual', 'Predicted', 'Error'],
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_prediction_analysis(y_true, y_pred, edge_names, save_dir=None):
    """Create comprehensive prediction analysis plots"""
    # Create directory for plots if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot ROC curve
    plot_roc_curve(y_true, y_pred, 
                  save_path=os.path.join(save_dir, 'roc_curve.png') if save_dir else None)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, 
                         save_path=os.path.join(save_dir, 'confusion_matrix.png') if save_dir else None)
    
    # Plot detailed predictions
    plot_detailed_predictions(y_true, y_pred, edge_names,
                            save_path=os.path.join(save_dir, 'detailed_predictions.png') if save_dir else None)
    
    # Print top predictions
    print("\nTop 5 Edge Predictions:")
    for i in range(min(5, len(y_pred))):
        print(f"{edge_names[i]}: Predicted = {y_pred[i]:.4f}, Actual = {y_true[i]:.4f}")

def visualize_graph_with_predictions(G, predictions, edge_weights, save_path=None):
    """Visualize the graph with edge weights and predictions"""
    plt.figure(figsize=(12, 8))
    
    # Create layout
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, alpha=0.6)
    
    # Draw edges with weights
    edge_colors = ['red' if abs(p - w) > 0.2 else 'green' 
                  for p, w in zip(predictions, edge_weights)]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                          width=2, alpha=0.6)
    
    # Add labels
    nx.draw_networkx_labels(G, pos)
    
    # Add edge labels
    edge_labels = {edge: f'{w:.2f} (pred: {p:.2f})' 
                  for (edge, w, p) in zip(G.edges(), edge_weights, predictions)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title('Graph with Edge Weights and Predictions')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 