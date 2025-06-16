# Biomarker GNN Demo

This project demonstrates the use of Graph Neural Networks (GNNs) for predicting heart disease risk based on patient biomarker data.

## Project Structure

- `src/`: Contains the source code for the project.
  - `data_processing.py`: Script for processing patient data.
  - `graph_builder.py`: Script for building the graph structure.
  - `gnn_model.py`: Contains the GNN model definition and training functions.
  - `inference.py`: Script for making predictions using the trained model.
  - `train.py`: Script for training the GNN model.
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