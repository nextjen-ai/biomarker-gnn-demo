# src/graph_builder.py
import pandas as pd
import networkx as nx
import numpy as np
import pickle
import matplotlib.pyplot as plt
from config import *

def visualize_graph(G, node_features, output_dir=DATA_DIR):
    """Visualize the knowledge graph"""
    print("Creating graph visualization...")
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Set up the layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    biomarker_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'biomarker']
    disease_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'disease']
    
    # Draw biomarker nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=biomarker_nodes,
        node_color='lightblue',
        node_size=2000,
        alpha=0.7,
        label='Biomarkers'
    )
    
    # Draw disease nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=disease_nodes,
        node_color='lightgreen',
        node_size=2000,
        alpha=0.7,
        label='Diseases'
    )
    
    # Draw edges with weights
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, pos,
        width=[w * 2 for w in edge_weights],
        alpha=0.5,
        edge_color='gray'
    )
    
    # Add labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_weight='bold'
    )
    
    # Add edge labels (weights)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8
    )
    
    # Add title and legend
    plt.title("Biomarker-Disease Knowledge Graph", pad=20)
    plt.legend()
    
    # Save the plot
    plt.savefig(
        os.path.join(output_dir, 'knowledge_graph.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    # Create a second visualization showing node features
    plt.figure(figsize=(15, 10))
    
    # Create a table of node features
    feature_data = []
    for node in G.nodes():
        features = node_features[node]
        if G.nodes[node]['type'] == 'biomarker':
            feature_data.append([
                node,
                'Biomarker',
                f"{features['mean']:.2f}",
                f"{features['std']:.2f}",
                f"{features['min']:.2f}",
                f"{features['max']:.2f}"
            ])
        else:
            feature_data.append([
                node,
                'Disease',
                f"{features['prevalence']:.2f}",
                f"{features['severity']:.2f}",
                '-',
                '-'
            ])
    
    # Create table
    plt.table(
        cellText=feature_data,
        colLabels=['Node', 'Type', 'Mean/Prevalence', 'Std/Severity', 'Min', 'Max'],
        loc='center',
        cellLoc='center'
    )
    
    # Remove axes
    plt.axis('off')
    
    # Add title
    plt.title("Node Features", pad=20)
    
    # Save the feature table
    plt.savefig(
        os.path.join(output_dir, 'node_features.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

def build_knowledge_graph():
    """Build the knowledge graph from processed data"""
    print("Building knowledge graph...")
    
    # Load processed data
    processed_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_heart_disease.csv'))
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add biomarker nodes
    biomarkers = [
        'trestbps',  # resting blood pressure
        'chol',      # serum cholesterol
        'thalach',   # maximum heart rate
        'oldpeak',   # ST depression
        'fbs',       # fasting blood sugar > 120 mg/dl
        'exang',     # exercise induced angina
        'slope',     # slope of peak exercise ST segment
        'ca'         # number of major vessels colored by flourosopy
    ]
    
    # Add disease node
    diseases = ['heart_disease']
    
    # Add nodes to graph
    for biomarker in biomarkers:
        G.add_node(biomarker, type='biomarker')
    
    for disease in diseases:
        G.add_node(disease, type='disease')
    
    # Calculate correlations and add edges
    for biomarker in biomarkers:
        # Calculate correlation with target (heart disease)
        correlation = abs(processed_data[biomarker].corr(processed_data['target']))
        
        # Add edge to heart disease
        G.add_edge(
            biomarker, 
            'heart_disease',
            weight=correlation,
            type='associated_with'
        )
    
    # Save graph structure using pickle
    with open(os.path.join(DATA_DIR, 'knowledge_graph.pkl'), 'wb') as f:
        pickle.dump(G, f)
    
    # Create node features
    node_features = {}
    
    # Add biomarker features
    for biomarker in biomarkers:
        node_features[biomarker] = {
            'mean': processed_data[biomarker].mean(),
            'std': processed_data[biomarker].std(),
            'min': processed_data[biomarker].min(),
            'max': processed_data[biomarker].max()
        }
    
    # Add disease features
    for disease in diseases:
        node_features[disease] = {
            'prevalence': processed_data['target'].mean(),
            'severity': 1.0
        }
    
    # Save node features
    pd.DataFrame(node_features).to_csv(
        os.path.join(DATA_DIR, 'node_features.csv')
    )
    
    return G, node_features

def analyze_graph(G):
    """Analyze the knowledge graph structure"""
    print("\nGraph Analysis:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # Analyze node types
    biomarker_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'biomarker']
    disease_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'disease']
    
    print(f"\nNode Types:")
    print(f"Biomarkers: {len(biomarker_nodes)}")
    print(f"Diseases: {len(disease_nodes)}")
    
    # Analyze edge weights
    edge_weights = [d['weight'] for _, _, d in G.edges(data=True)]
    print(f"\nEdge Weights:")
    print(f"Mean: {np.mean(edge_weights):.3f}")
    print(f"Std: {np.std(edge_weights):.3f}")
    print(f"Min: {np.min(edge_weights):.3f}")
    print(f"Max: {np.max(edge_weights):.3f}")

def main():
    """Main function to build and analyze the knowledge graph"""
    print("Starting knowledge graph construction...")
    
    try:
        # Build graph
        G, node_features = build_knowledge_graph()
        
        # Analyze graph
        analyze_graph(G)
        
        # Visualize graph
        visualize_graph(G, node_features)
        
        print("\nKnowledge graph construction complete!")
        
    except Exception as e:
        print(f"Error in knowledge graph construction: {str(e)}")
        raise

if __name__ == "__main__":
    main()