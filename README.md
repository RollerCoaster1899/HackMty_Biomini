
==============================================================================
AUTOSCREEN - ACCELERATING DRUG DISCOVERY PIPELINE
==============================================================================
AutoScreen implements an end-to-end drug discovery workflow powered by
Biomini Team. The application streamlines the transition from target
identification to lead optimization:
Data Acquisition: Direct querying of the ChEMBL database for specific
protein targets and IC50 bioactivity data.
Preprocessing: Automated cleaning of SMILES, salt removal, and
conversion of IC50 to pIC50 values.
Feature Engineering: Simultaneous calculation of Lipinski physicochemical
descriptors and 2048-bit ECFP4 Morgan fingerprints.
Model Benchmarking: Comparison of 7 regression algorithms to identify
the optimal predictor for the specific target.
Lead Optimization: Automated generation of molecular mutants using
SELFIES to enhance potency beyond the initial screening library.
Visualization: 3D molecular embedding and GIF generation for structural
analysis of lead candidates.

2. PROJECT STRUCTURE
AutoScreen/
|-- app.py [Main Streamlit application]
|-- world.csv [Initial screening library]
|-- biomini_gif.gif [UI Logo]
|-- WORKFLOW_AUTOSCREEN.jpeg [Methodology diagram]
|
|-- best_model.pkl [Saved artifact of the top regressor]
|-- *.gif [Generated 3D rotating molecule files]
|
|-- Data_Outputs/
| |-- *_S3_bioactivity_data.csv [Raw ChEMBL data]
| |-- *_S4_bioactivity_data.csv [Cleaned data with descriptors]
| |-- predictions.csv [Potency results for world.csv]
| |-- best_mutants_table.csv [Optimized molecular candidates]
| |-- predicted_molecules.csv [Results from user-uploaded files]
==============================================================================
3. KEY FEATURES
[ Data Processing ]
ChEMBL Client: Real-time API integration for protein target search.
Bioactivity Filtering: Automatic extraction of IC50 and unit normalization.
Molecular Standardization: SMILES canonicalization via RDKit.
[ Molecular Representations ]
Lipinski Rule of 5: MolWt, LogP, H-Bond Donors, and H-Bond Acceptors.
Fingerprints: 2048-bit Morgan Fingerprints (ECFP4) with radius 2.
[ Machine Learning Models ]
Regressors: Random Forest, Linear Regression, SVR, Decision Trees,
Gradient Boosting, AdaBoost, and K-Nearest Neighbors.
Automated Selection: Automatic deployment of the model with the lowest MSE.
[ Lead Optimization & Visualization ]
SELFIES Mutation: Robust molecular string mutation ensuring chemical validity.
3D Engine: RDKit AllChem embedding with Matplotlib 3D projection.
GIF Animation: Rotating view generation for structural inspection.
==============================================================================
4. INSTALLATION
Requirements: Python 3.9 - 3.11
Step 1: Create Environment
conda create -n autoscreen python=3.10
conda activate autoscreen
Step 2: Install Dependencies
pip install streamlit pandas chembl_webresource_client rdkit scikit-learn
pip install numpy matplotlib seaborn joblib selfies pillow scipy
Step 3: Run Application
streamlit run app.py
==============================================================================
5. USAGE
Step-by-Step Workflow
Target Search: Enter a protein name (e.g., "EGFR") to retrieve ChEMBL IDs.
Training: Select a target. The app will fetch data and train models.
Analysis: View the Lipinski distribution and Model performance table.
Screening: The app automatically screens "world.csv" and saves predictions.
Mutation: Review the "Generating Drug Mutation" section for optimized
structures and 3D animations.
Custom Upload: Use the file uploader at the bottom to predict potencies
for your own .csv library (requires 'smiles' and 'zinc_id' columns).
==============================================================================
6. CONFIGURATION OPTIONS
Internal Parameters (Adjustable in app.py):
ECFP4 nBits: Default set to 2048.
Mutation Rate: Default set to 3 mutations per SELFIES string.
Mutant Count: Generates 100 valid mutants per top candidate.
Test Split: 20% of data held for model validation.
GIF Frames: 30 frames per rotation for visual clarity.
==============================================================================
7. OUTPUT FILES AND METRICS
Performance Metrics
The app generates a performance table containing:
MAE (Mean Absolute Error)
MSE (Mean Squared Error)
R^2 (Coefficient of Determination)
Optimization Results
File: best_mutants_table.csv
Contains the "Original SMILES" and its corresponding mutants that exhibited
increased predicted pIC50 values.
Visual Artifacts
Rotating GIFs (1.gif, 2.gif, etc.) represent the 3D structures of the
original lead compound and its optimized mutant counterparts.
==============================================================================
8. TROUBLESHOOTING
Issue: "Invalid SMILES" warning
Solution: Ensure the input CSV uses standard canonical SMILES. The pipeline
will automatically skip molecules that RDKit cannot parse.
Issue: ChEMBL API Timeout
Solution: If the database is busy, the app may stall during download.
Restart the query or check your internet connection.
Issue: GIF generation is slow
Solution: 3D embedding is CPU intensive. Reduce the 'num_frames' or
'num_mutants' in the source code to speed up processing.
