# Biomarker-Disease Prediction Demo

A demonstration project for predicting disease risks using biomarker data and Graph Neural Networks (GNNs).

## Project Overview

This demo project shows how to:
1. Process biomarker data from the UCI Heart Disease dataset
2. Build a knowledge graph connecting biomarkers and diseases
3. Train a GNN to predict disease risks
4. Make predictions for new patients

## Features

### Data Processing
- ✅ Process heart disease dataset
- ✅ Extract relevant biomarkers
- ✅ Normalize and prepare data for GNN

### Knowledge Graph
- ✅ Build graph connecting biomarkers and diseases
- ✅ Calculate edge weights based on correlations
- ✅ Store node features and graph structure

### GNN Model
- ✅ Implement graph neural network (GCN) architecture
- ✅ Train model to predict disease risks
- ✅ Evaluate model performance
- ✅ Save trained model for inference

### Inference
- ✅ Load trained model
- ✅ Process new patient data
- ✅ Predict disease risks
- ✅ Output risk percentages

## Setup and Usage

1. Create and activate virtual environment:
```bash
uv sync
```

2. Run the pipeline in sequence:

   a. Process the data:
   ```bash
   python src/data_processing.py
   ```
   This will download and process the heart disease dataset.

   b. Build the knowledge graph:
   ```bash
   python src/graph_builder.py
   ```
   This creates the graph structure and node features.

   c. Train the GNN model:
   ```bash
   python src/gnn_model.py
   ```
   This trains the model and saves it for inference.

   d. Make predictions for a new patient:
   ```bash
   python src/inference.py
   ```
   This loads the trained model and makes predictions.

Note: Each step must be run in order as they depend on the output of the previous step.

## Project Structure

```
.
├── data/                      # Data files
│   ├── biomarker_nodes.csv    # Generated node data
│   ├── disease_nodes.csv      # Generated node data
│   ├── heart_disease.csv      # Original dataset
│   ├── node_features.csv      # Generated node feature data
│   ├── processed_heart_disease.csv      # Generated data
│   ├── relationships.csv      # Generated data
│   ├── knowledge_graph.pkl    # Built knowledge graph
│   └── gnn_model.pt           # Trained GNN model
├── src/                       # Source code
│   ├── config.py              # Configuration
│   ├── data_processing.py     # Data processing pipeline
│   ├── graph_builder.py       # Knowledge graph construction
│   ├── gnn_model.py           # GNN model and training
│   └── inference.py           # Inference for new patients
└── pyproject.toml             # Project configuration
```

## Biomarkers and Diseases

The model works with the following biomarkers:
- trestbps (resting blood pressure)
- chol (serum cholesterol)
- thalach (maximum heart rate)
- oldpeak (ST depression)

And predicts risks for:
- Heart disease
- Hypertension
- High cholesterol
- Angina

## Making Predictions

To make predictions for a new patient, provide their biomarker values:
```python
patient_data = {
    'trestbps': 145,  # resting blood pressure
    'chol': 250,      # serum cholesterol
    'thalach': 150,   # maximum heart rate
    'oldpeak': 1.5    # ST depression
}
```

The model will output risk percentages for each disease.

## Data Sources

- UCI Heart Disease Dataset
  - Contains biomarker measurements
  - Includes disease presence/absence
  - Clean, processed data

## Contributing

This is a demo project. Feel free to fork and experiment with different approaches.

## License

MIT License 