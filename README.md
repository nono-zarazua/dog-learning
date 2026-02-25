# Dog vs Chicken — Logistic Regression (Baseline)

This branch implements a complete binary image classification pipeline (dog vs chicken) using handcrafted features and Logistic Regression.

The notebook is designed to be fully reproducible and portable, using paths relative to the repository root.

---

## Project Overview

The goal of this project is to build a lightweight and interpretable baseline classifier before exploring more complex approaches.

The pipeline follows these main steps:

1. Load images from structured folders  
2. Extract numerical features from each image  
3. Train a Logistic Regression classifier  
4. Evaluate model performance  
5. Generate predictions and export results  

---

## Dataset Structure

The expected folder structure is:

data/
└── dog_chicken/
├── train/
│   ├── dog/
│   └── chicken/
└── test/
├── dog/
└── chicken/

- `train/` contains images used for model training.
- `test/` contains unseen images used for final evaluation.
- Each class folder contains `.jpg`, `.jpeg`, or `.png` images.

---

# Step-by-Step Pipeline Description

## Step 1 — Imports and Project Paths

All required libraries are imported, and dataset paths are defined relative to the repository root.

Using relative paths ensures that the notebook runs correctly after cloning the repository on any machine.

---

## Step 2 — Collect Image Paths

The notebook scans the dataset directories and collects file paths for:

- Training dog images  
- Training chicken images  
- Test dog images  
- Test chicken images  

This creates structured lists of images that will be processed during feature extraction.

---

## Step 3 — Feature Extraction

Each image is:

- Loaded as RGB
- Resized to a fixed resolution
- Normalized to [0, 1]

From each image, a **20-dimensional feature vector** is extracted, consisting of:

- RGB mean and standard deviation (6 features)
- Grayscale mean and standard deviation (2 features)
- Gradient magnitude statistics (2 features)
- 10-bin grayscale histogram (10 features)

These handcrafted features provide a compact global representation of each image.

---

## Step 4 — Build Training Feature Table

Features are extracted for all training images and stored in a structured DataFrame.

Labels are assigned as:

- Dog → 1
- Chicken → 0

Each row corresponds to one image, and each column corresponds to a numerical feature.

---

## Step 5 — Data Sanity Checks

Before training, we verify:

- No missing values
- Correct data types
- Balanced class distribution

These checks ensure data integrity and prevent silent errors.

---

## Step 6 — Define Features and Target

The dataset is separated into:

- **X** → Feature matrix (20 numerical features)
- **y** → Binary target labels

Metadata columns such as filename and path are excluded from training.

---

## Step 7 — Model Training and Cross-Validation

A scikit-learn Pipeline is defined:

- `StandardScaler()` for normalization
- `LogisticRegression()` for classification

Performance is evaluated using:

- Stratified 5-fold cross-validation
- F1-score as evaluation metric

This provides a robust estimate of generalization performance.

---

## Step 8 — Final Model Training and Saving

After cross-validation, the model is retrained using the full training dataset.

The trained pipeline is saved as:
    logreg_dog_vs_chicken.joblib

This allows the model to be reused without retraining.

---

## Step 9 — Test Evaluation and Prediction Export

Features are extracted from the test dataset and passed through the trained model.

We compute:

- Accuracy
- F1-score
- Classification report

Predictions and probabilities are exported to:
    test_predictions.csv

---

## Step 10 — Reproducibility Information

The notebook prints:

- Python version
- scikit-learn version
- numpy version
- pandas version

This ensures reproducibility across different environments.

---

# Example Performance

- Cross-validation mean F1 ≈ 0.88  
- Test Accuracy ≈ 0.79  
- Test F1 ≈ 0.77  

The model performs well as a simple baseline, though some confusion remains between visually similar dog and chicken images due to the use of global handcrafted features.

---

# Repository Contents (logistic branch)

- `Logistic.ipynb` — Full training and evaluation notebook  
- `feature_extraction.py` — Additional feature extraction script  
- `data/` — Dataset folders  
- `logreg_dog_vs_chicken.joblib` — Saved trained model  
- `test_predictions.csv` — Exported test predictions  

---

# Quick Start

## 1) Clone the repository

```bash
git clone https://github.com/nono-zarazua/dog-learning.git
cd dog-learning
git checkout logistic

## 2) Create an environment
conda create -n bioimage_ml python=3.11 -y
conda activate bioimage_ml
pip install numpy pandas scikit-learn pillow joblib jupyter

## 3) Run the notebook 
jupyter notebook

Open Logistic.ipynb and execute all cells sequentially.

# Author
Alba de Prada Hernández

