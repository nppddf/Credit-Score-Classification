# Credit Score Classification

A machine learning project for classifying credit scores using various classification algorithms. This project includes data preprocessing, exploratory data analysis, model training, and evaluation pipelines.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Notebooks](#notebooks)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to build and evaluate machine learning models for credit score classification. The system processes credit-related features to predict creditworthiness, which can be used by financial institutions for risk assessment and decision-making.

## Features

- **Data Download**: Automated dataset download from Kaggle
- **Data Preprocessing**: Comprehensive data cleaning and feature engineering
- **Exploratory Data Analysis**: Interactive notebooks for data exploration
- **Model Training**: Multiple classification algorithms for comparison
- **Model Evaluation**: Performance metrics and visualization

## Project Structure

```
Credit-Score-Classification/
├── data/                      # Dataset directory (created after download)
├── notebooks/                 # Jupyter notebooks
│   ├── eda.ipynb             # Exploratory Data Analysis
│   └── modeling.ipynb        # Model development and evaluation
├── src/                       # Source code
│   ├── download_dataset.py   # Dataset download script
│   ├── preprocessing.py      # Data preprocessing utilities
│   └── train.py              # Model training script
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
├── LICENSE                   # License file
└── README.md                 # Project documentation
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Kaggle API credentials (for dataset download)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Credit-Score-Classification.git
   cd Credit-Score-Classification
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Kaggle API credentials**
   - Sign up for a Kaggle account at https://www.kaggle.com/
   - Go to your account settings and create an API token
   - Download `kaggle.json` and place it in `~/.kaggle/` directory
   - Alternatively, set `KAGGLE_API_TOKEN` environment variable
   - Copy `.env.example` to `.env` and add your credentials:
     ```bash
     cp .env.example .env
     ```

## Usage

### Download Dataset

Download the credit scoring dataset from Kaggle:

```bash
python src/download_dataset.py
```

The dataset will be downloaded to the `data/` directory.

### Run Exploratory Data Analysis

Open and run the EDA notebook:

```bash
jupyter notebook notebooks/eda.ipynb
```

### Preprocess Data

Run the preprocessing script:

```bash
python src/preprocessing.py
```

### Train Models

Train classification models:

```bash
python src/train.py
```

### Use Jupyter Notebooks

For interactive development and analysis:

```bash
jupyter notebook notebooks/
```

## Dataset

The project uses the Credit Scoring Dataset from Kaggle:
- **Dataset**: `maksimkotenkov/credit-scoring-dataset`
- **Source**: [Kaggle](https://www.kaggle.com/datasets/maksimkotenkov/credit-scoring-dataset)

### Dataset Description

[Add description of the dataset features, target variable, and data characteristics here]

## 📓 Notebooks

### Exploratory Data Analysis (`notebooks/eda.ipynb`)

This notebook contains:
- Data loading and initial inspection
- Statistical summaries
- Data visualization
- Feature analysis
- Missing value analysis
- Correlation analysis

### Modeling (`notebooks/modeling.ipynb`)

This notebook includes:
- Model selection and comparison
- Hyperparameter tuning
- Model training and evaluation
- Performance metrics visualization
- Feature importance analysis

## Model Training

The training pipeline includes:

1. **Data Loading**: Load and validate the dataset
2. **Preprocessing**: Handle missing values, encode categorical features, scale numerical features
3. **Feature Engineering**: Create new features if needed
4. **Model Selection**: Compare multiple algorithms (e.g., Logistic Regression, Random Forest, XGBoost)
5. **Training**: Train selected models with cross-validation
6. **Evaluation**: Calculate metrics (accuracy, precision, recall, F1-score, ROC-AUC)
7. **Saving**: Save trained models for future use

## Results

[Add model performance results, metrics, and visualizations here]

Example metrics to include:
- Accuracy scores
- Precision, Recall, F1-score
- ROC-AUC scores
- Confusion matrices
- Feature importance plots

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.