# Biomarker GNN Demo

A demonstration of using Graph Neural Networks (GNNs) for heart disease prediction based on biomarker data.

## Model Evolution

### Version 1.0 (Original Model)
- Simple GNN architecture with single feature per node
- Basic normalization of biomarker values
- Binary edge weights (1.0 for disease present, 0.5 for absent)
- Achieved 98.50% ± 0.71% accuracy in cross-validation
- Limitations:
  - No utilization of domain knowledge
  - Overly simplistic data generation
  - Potential overfitting to synthetic data

### Version 2.0 (Knowledge Graph Integration)
- Enhanced GNN architecture with three features per node:
  - Normalized biomarker value
  - Mean value from knowledge graph
  - Standard deviation from knowledge graph
- Correlation-based edge weights from knowledge graph
- Integration with domain knowledge through graph structure
- Current performance: 51.50% ± 4.89% accuracy in cross-validation

#### Key Changes in v2.0
1. **Knowledge Graph Integration**
   - Added graph structure based on biomarker correlations
   - Incorporated statistical features from real data
   - Used correlation values for edge weights

2. **Enhanced Feature Representation**
   - Each node now has 3 features instead of 1
   - Features include normalized value and statistical properties
   - Edge weights reflect actual correlations between biomarkers

3. **Model Architecture**
   - Same GNN structure but with wider input features
   - Maintained two GCN layers with ReLU activation
   - Added dropout for regularization

#### Current Challenges
1. **Performance Degradation**
   - New model performs worse than the original (51.50% vs 98.50%)
   - High variance in cross-validation results (±4.89%)
   - Model struggles to learn from enhanced features

2. **Potential Issues**
   - Mismatch between knowledge graph structure and synthetic data generation
   - Statistical features may not be informative for individual predictions
   - Complex feature representation may be harder to learn from

3. **Future Improvements Needed**
   - Align data generation with knowledge graph relationships
   - Evaluate feature importance and potentially simplify
   - Test with real patient data instead of synthetic data
   - Consider alternative graph structures or feature representations

## Project Structure

- `src/`: Contains the source code for the project.
  - `data_processing.py`: Script for processing patient data.
  - `graph_builder.py`: Script for building the graph structure.
  - `gnn_model.py`: Contains the GNN model definition and training functions.
  - `inference.py`: Script for making predictions using the trained model.
  - `train.py`: Script for training the GNN model.
  - `evaluate_models.py`: Script for comparing model versions.
  - `config.py`: Contains configuration parameters.
- `data/`: Contains the processed data and the saved model.
  - `model.pth`: The trained GNN model saved in this directory.

## Usage

1. **Data Processing**:
   ```bash
   uv run src/data_processing.py
   ```

2. **Graph Building**:
   ```bash
   uv run src/graph_builder.py
   ```

3. **Training the Model**:
   ```bash
   uv run src/train.py
   ```

4. **Making Predictions**:
   ```bash
   uv run src/inference.py
   ```

5. **Comparing Model Versions**:
   ```bash
   uv run src/evaluate_models.py
   ```

## Dependencies
- PyTorch
- PyTorch Geometric
- NetworkX
- NumPy
- scikit-learn

## Future Work
1. Improve data generation to better reflect knowledge graph relationships
2. Experiment with different feature representations
3. Test on real patient data
4. Explore alternative graph structures
5. Implement feature importance analysis

## Model

The GNN model is defined in `src/gnn_model.py`. It uses a graph structure to represent the relationships between biomarkers and heart disease. The model is trained using the data processed in `data_processing.py` and the graph structure built in `graph_builder.py`.

## Inference

The `inference.py` script loads the trained model from `data/model.pth` and uses it to predict heart disease risk for new patient data.

## Project Overview

This project aims to predict the risk of heart disease based on patient biomarker data. The model is trained to identify patterns in the data that may indicate a higher risk of heart disease.

## Features

- **Data Processing**: Processes patient biomarker data for use in the GNN model.
- **Graph Building**: Constructs a knowledge graph connecting biomarkers and heart disease.
- **Model Training**: Trains a GNN to predict heart disease risk.
- **Inference**: Makes predictions for new patients based on their biomarker data.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.