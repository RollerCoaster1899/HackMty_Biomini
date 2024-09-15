# SMILES to pIC50 Prediction Pipeline

This repository provides a pipeline for predicting pIC50 values of molecules from their SMILES representations. The pipeline involves data retrieval, cleaning, feature extraction, model training, and prediction, with additional functionality for mutating SMILES strings and visualizing results.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Usage](#usage)
    - [Step 1: Retrieve Target Protein Data](#step-1-retrieve-target-protein-data)
    - [Step 2: Select Target Protein](#step-2-select-target-protein)
    - [Step 3: Download Bioactivity Data](#step-3-download-bioactivity-data)
    - [Step 4: Clean Bioactivity Data](#step-4-clean-bioactivity-data)
    - [Step 5: Calculate Bioactivity Classes](#step-5-calculate-bioactivity-classes)
    - [Step 6: Feature Extraction](#step-6-feature-extraction)
    - [Step 7: Plot Distributions](#step-7-plot-distributions)
    - [Step 8: Train and Evaluate Models](#step-8-train-and-evaluate-models)
    - [Step 11: Predict with Best Model](#step-11-predict-with-best-model)
    - [Step 12: Predict Single Molecule](#step-12-predict-single-molecule)
    - [Step 13: Generate Mutants](#step-13-generate-mutants)
3. [Files](#files)
4. [Contributing](#contributing)
5. [License](#license)

## Prerequisites

Ensure the following Python packages are installed:
- pandas
- rdkit
- scikit-learn
- seaborn
- matplotlib
- joblib
- selfies

You can install them using pip:
```bash
pip install pandas rdkit scikit-learn seaborn matplotlib joblib selfies
```

## Usage

### Step 1: Retrieve Target Protein Data
Input the target protein name to search for the target data. The data is retrieved and displayed.

```python
target_protein_name = input("Put the target: ")
```

### Step 2: Select Target Protein
Choose a target protein from the retrieved list by its index. The selected target's ID is displayed.

```python
selected_target_index = int(input("Put the selected target: "))
```

### Step 3: Download Bioactivity Data
Download bioactivity data for the selected target protein, focusing on IC50 values, and save it as a CSV file.

```python
def Download_bioctivity_data(selected_target):
    # Implementation
```

### Step 4: Clean Bioactivity Data
Clean the downloaded data by filtering out incomplete and duplicate entries, and save the cleaned data.

```python
def Cleaning_bioactivity_data(selected_target, data):
    # Implementation
```

### Step 5: Calculate Bioactivity Classes
Prepare the data for modeling by saving bioactivity classes into a CSV file.

```python
def Calculate_bioactivity_classes(selected_target, data):
    # Implementation
```

### Step 6: Feature Extraction
Extract Lipinski descriptors and ECFP4 fingerprints from the SMILES strings and compute pIC50 values.

```python
def apply_lipinski_and_pIC50(df):
    # Implementation
```

### Step 7: Plot Distributions
Visualize the distribution of Lipinski descriptors.

```python
def plot_lipinski_distribution(selected_target, df):
    # Implementation
```

### Step 8: Train and Evaluate Models
Train various regression models, evaluate their performance, and save the best model based on MSE.

```python
def train_and_evaluate_model(model, model_name, X_train, X_test, Y_train, Y_test):
    # Implementation
```

### Step 11: Predict with Best Model
Load the best model, make predictions on new data, and save the predictions to a CSV file.

```python
def predict_molecules_from_csv(input_csv, output_csv, smiles_col, id_col=None):
    # Implementation
```

### Step 12: Predict Single Molecule
Predict the pIC50 value for a single molecule given its SMILES string.

```python
def predict_single_molecule(smiles, model):
    # Implementation
```

### Step 13: Generate Mutants
Generate and evaluate mutants of a given SMILES string, and visualize them as 3D GIFs.

```python
def generate_valid_mutants(smiles, num_mutants, smiles_symbols, max_attempts=1000):
    # Implementation
```

## Files

- `pipeline.py`: Contains the core implementation of the SMILES to pIC50 prediction pipeline.
- `best_mutants_table.csv`: CSV file with the best mutants and their predicted pIC50 values.
- `Predictions.csv`: CSV file with predictions for new molecules.
- `*.gif`: Generated GIFs for 3D visualizations of mutants.

## Contributing

Feel free to submit issues or pull requests to improve the functionality or fix bugs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
