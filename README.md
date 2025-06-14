# Biomarker-Disease Prediction Demo

A demonstration project for predicting disease risks using biomarker data and Graph Neural Networks (GNNs).

## Project Overview

This demo project shows how to:
1. Process biomarker data from the UCI Heart Disease dataset
2. Build a knowledge graph connecting biomarkers and diseases
3. Visualize the relationships between biomarkers and diseases
4. (Coming soon) Train a GNN to predict disease risks

## Current Progress

### Completed
- ✅ Data processing pipeline for heart disease dataset
- ✅ Knowledge graph construction
- ✅ Graph visualization
- ✅ Node feature analysis

### In Progress
- 🔄 GNN model development
- 🔄 Training pipeline
- 🔄 Prediction system

## Setup

Simply run:
```bash
uv sync
```

This will:
- Create a virtual environment
- Install all dependencies
- Create a lockfile for reproducibility

## Project Structure

```
.
├── data/                      # Data files
│   ├── heart_disease.csv      # Original dataset
│   └── processed/             # Processed data files
├── src/                       # Source code
│   ├── config.py              # Configuration
│   ├── data_processing.py     # Data processing pipeline
│   └── graph_builder.py       # Knowledge graph construction
└── pyproject.toml     # Project configuration
```

## Next Steps

1. GNN Model Development
   - Implement graph neural network architecture
   - Add node and edge feature processing
   - Set up training pipeline

2. Training and Evaluation
   - Train the model on the knowledge graph
   - Implement evaluation metrics
   - Add cross-validation

3. Prediction System
   - Create prediction interface
   - Add confidence scoring
   - Implement visualization of predictions

## Data Sources

- UCI Heart Disease Dataset
  - Contains biomarker measurements
  - Includes disease presence/absence
  - Clean, processed data

## Visualization

The project includes two types of visualizations:
1. Knowledge Graph Visualization
   - Shows relationships between biomarkers and diseases
   - Edge weights indicate relationship strength
   - Different colors for different node types

2. Node Feature Analysis
   - Statistical properties of biomarkers
   - Disease prevalence and severity
   - Feature importance analysis

## Contributing

This is a demo project. Feel free to fork and experiment with different approaches.

## License

MIT License 