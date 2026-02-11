# Neural Network Tabular Data Training Library (nntab_v1)

A comprehensive Python package for training neural networks on tabular data with support for curriculum learning, confidence-aware training, and extensive logging capabilities.

## Table of Contents
- [Process Flow](#process-flow)
- [Module Documentation](#module-documentation)
- [Configuration Guide](#configuration-guide)
- [Usage Examples](#usage-examples)
- [Output Structure](#output-structure)

## Process Flow

The training pipeline follows this sequential workflow:

```
1. Configuration Loading (config_loader.py)
   ↓
2. Logger Initialization (logger.py)
   ↓
3. Data Loading & Preprocessing (datasets/)
   ↓
4. Model Initialization (models/)
   ↓
5. Training Execution (utils/)
   ↓
6. Visualization & Results (plots/)
```

### Detailed Flow:

1. **Initialization**: `main.py` loads configuration from `config.json`
2. **Logging Setup**: Creates experiment-specific log files in `/logs/`
3. **Data Pipeline**: Loads and preprocesses training/validation data
4. **Model Creation**: Initializes neural network architecture with optimizers
5. **Training Loop**: Executes chosen training method (standard/curriculum learning)
6. **Model Persistence**: Saves best models to `/Models/{exp_name}/{model_name}/`
7. **Visualization**: Generates loss curves in `/plots/{exp_name}/{model_name}/`

## Module Documentation

### 📁 `datasets/`
**Purpose**: Data loading, preprocessing, and dataset management

#### Files:
- **`dataset.py`**: Core dataset classes and data loading functionality
- **`preprocess.py`**: Data preprocessing utilities and transformations
- **`dataloading_helpers.py`**: Helper functions for data manipulation

#### Key Functions:
- `read_train_data()`: Loads training and validation data from specified paths
- Data preprocessing with feature scaling and encoding
- PyTorch DataLoader creation with configurable batch sizes

### 📁 `models/`
**Purpose**: Neural network architecture definitions and model initialization

#### Files:
- **`model.py`**: Neural network architecture implementations
- **`model_initialise.py`**: Model initialization, optimizer, and scheduler setup

#### Key Functions:
- `get_model()`: Creates model instance based on configuration
- `fraudmodel_5layer()`: Specific 5-layer neural network architecture
- Optimizer and learning rate scheduler initialization
- Loss function setup (classification, ranking for curriculum learning)

### 📁 `utils/`
**Purpose**: Training algorithms, evaluation metrics, and utility functions

#### Files:
- **`utils.py`**: Main training functions and evaluation metrics
- **`calculate_scores.py`**: Advanced scoring metrics (AUM, EL2N, forgetting scores)
- **`crl_utils.py`**: Curriculum learning utilities and history tracking

#### Key Functions:
- `train_model()`: Standard training with early stopping
- `train_model_crl()`: Curriculum learning training pipeline
- `train_fn()`: Single epoch training function
- `val_fn()`: Validation evaluation function
- `train_crl()`: Curriculum learning training with confidence ranking
- `check_for_invalid_values()`: Tensor validation for NaN/inf values

### 📁 `plots/`
**Purpose**: Visualization and results analysis

#### Files:
- **`plots.py`**: Plotting functions and visualization utilities

#### Key Functions:
- `plot_loss_curve()`: Training/validation loss visualization
- `auc_plot()`: ROC-AUC and classification metrics plotting
- `plot_info()`: Business KPI analysis and threshold optimization

### 🔧 Core Files:
- **`main.py`**: Entry point and orchestration
- **`config_loader.py`**: Configuration file handling
- **`logger.py`**: Comprehensive logging system
- **`config.json`**: Training configuration parameters

## Configuration Guide

### config.json Parameters

#### **Data Configuration**
```json
{
    "train_path": "path/to/training/data.csv",
    "test_path": "path/to/validation/data.csv", 
    "feature_path": "path/to/feature/definitions.json",
    "target": "fraud_sw",
    "identifier_cols": ["transaction_id", "customer_id"]
}
```
- **train_path**: Path to training dataset (CSV format)
- **test_path**: Path to validation dataset (CSV format)
- **feature_path**: Path to feature definitions (optional)
- **target**: Target column name for prediction
- **identifier_cols**: List of column names to create unique identifiers (required for confidence_aware training)

#### **Model Architecture**
```json
{
    "model_name": "fraudmodel_5layer",
    "hidden_layers": [512, 128, 64, 4],
    "output_dim": 2,
    "batch_size": 512,
    "num_workers": 0
}
```
- **model_name**: Model identifier for saving (e.g., "fraudmodel_5layer", "transformer_model")
- **hidden_layers**: List of hidden layer sizes [input_size, hidden1, hidden2, ..., final_hidden]
- **output_dim**: Number of output classes (2 for binary, N for multiclass)
- **batch_size**: Training batch size (suggested: 256-1024)
- **num_workers**: DataLoader workers (0 for single-threaded)

#### **Training Configuration**
```json
{
    "epochs": 100,
    "training_method": "standard_multiclass",
    "use_class_weights": true,
    "optimizer": {
        "name": "Adam",
        "lr": 1e-4,
        "weight_decay": 1e-5
    }
}
```
- **epochs**: Maximum training epochs (suggested: 50-200)
- **training_method**: Training approach
  - `"standard_multiclass"`: Standard classification training
  - `"standard_multilabel"`: Multi-label classification
  - `"confidence_aware"`: Curriculum learning with confidence ranking
- **use_class_weights**: Balance class weights automatically
- **optimizer**: Optimizer configuration
  - **name**: "Adam", "SGD", "AdamW"
  - **lr**: Learning rate (suggested: 1e-4 to 1e-3)
  - **weight_decay**: L2 regularization (suggested: 1e-5 to 1e-4)

#### **Curriculum Learning (Advanced)**
```json
{
    "training_method": "confidence_aware",
    "identifier_cols": ["transaction_date", "customer_id", "sequence_number"],
    "rank_weight_c": 0.5,
    "rank_weight_f": 0.5,
    "health_metrics": ["aum", "el2n", "forgetting"]
}
```
- **training_method**: Must be "confidence_aware" for curriculum learning
- **identifier_cols**: List of column names to create unique sample identifiers (required for confidence_aware training)
  - Single column: `["transaction_id"]`
  - Multiple columns: `["date", "customer_id", "sequence"]` (concatenated with underscores)
- **rank_weight_c**: Confidence ranking loss weight (0.0-1.0)
- **rank_weight_f**: Forgetting ranking loss weight (0.0-1.0)
- **health_metrics**: Training dynamics metrics to track

#### **Experiment Management**
```json
{
    "exp_name": "fraud_detection_2024"
}
```
- **exp_name**: Experiment identifier for organizing outputs

## Usage Examples

### Command Line Execution
```bash
# Run with default config.json
python main.py

# Run from Jupyter notebook
from nntab_v1.main import main
main()  # Uses config.json

# Run with custom config
custom_config = {
    "exp_name": "my_experiment",
    "training_method": "confidence_aware",
    "epochs": 50
}
main(custom_config)
```

### Configuration Examples

#### Binary Classification (Standard)
```json
{
    "train_path": "data/fraud_train.csv",
    "test_path": "data/fraud_val.csv",
    "target": "is_fraud",
    "model_name": "binary_classifier",
    "hidden_layers": [256, 128, 64],
    "output_dim": 2,
    "training_method": "standard_multiclass",
    "batch_size": 512,
    "epochs": 100,
    "exp_name": "fraud_detection_baseline"
}
```

#### Curriculum Learning Setup
```json
{
    "training_method": "confidence_aware",
    "identifier_cols": ["transaction_date", "account_id", "sequence_nbr"],
    "rank_weight_c": 0.3,
    "rank_weight_f": 0.7,
    "model_name": "crl_model",
    "exp_name": "curriculum_learning_experiment"
}
```

#### Identifier Column Configuration
```json
{
    "identifier_cols": ["primary_key"]  // Single identifier column
}
```
```json
{
    "identifier_cols": ["date", "customer_id", "transaction_seq"]  // Multiple columns concatenated
}
```
**Note**: For `confidence_aware` training, identifier columns are used to track individual samples across epochs for curriculum learning. The system creates a unique identifier by concatenating the specified columns with underscores (e.g., "2023-01-01_12345_001").

## Output Structure

The package creates organized output directories:

```
project_root/
├── logs/
│   └── {exp_name}_{timestamp}.log
├── Models/
│   └── {exp_name}/
│       └── {model_name}/
│           └── {model_name}.pth
├── plots/
│   └── {exp_name}/
│       └── {model_name}/
│           └── {model_name}_loss.jpg
└── *.pkl (training dynamics files - AUM, EL2N, forgetting scores)
```

### Output Files:
- **Log Files**: Comprehensive training logs with timestamps
- **Model Files**: Best model checkpoints in PyTorch format
- **Plot Files**: Loss curves and training visualizations
- **Pickle Files**: Training dynamics and scoring metrics (curriculum learning)

## Logging System

The package includes comprehensive logging that works in both terminal and Jupyter environments:

- **Console Output**: Real-time training progress
- **File Logging**: Persistent logs with timestamps and function details
- **Jupyter Friendly**: Clean output format in notebook cells
- **Automatic Detection**: Adapts format based on execution environment

## Advanced Features

### Curriculum Learning
- **Confidence-aware training**: Models learn easier examples first
- **Forgetting dynamics**: Tracks which examples are frequently misclassified
- **Training health metrics**: AUM, EL2N, and forgetting scores

### Robust Training
- **Early stopping**: Prevents overfitting with validation monitoring
- **NaN/Inf detection**: Automatic detection of training instabilities
- **Class balancing**: Automatic handling of imbalanced datasets
- **Learning rate scheduling**: Adaptive learning rate adjustment

## Requirements

- Python 3.7+
- PyTorch
- scikit-learn
- matplotlib
- pandas
- numpy
- tqdm

## Quick Start

1. Configure your experiment in `config.json`
2. Place your data files in the specified paths
3. Run `python main.py`
4. Monitor progress in logs and console output
5. Find results in `Models/` and `plots/` directories

---

For additional support or questions, refer to the comprehensive logging output which provides detailed information about each step of the training process.
