                              A U T O S C R E E N
                     Accelerating Drug Discovery Pipeline
================================================================================
                           Powered by Biomini Team
================================================================================


TABLE OF CONTENTS
-----------------
    1. Overview
    2. Project Structure
    3. Key Features
    4. Installation
    5. Usage Guide
    6. Configuration
    7. Output Files & Metrics
    8. Troubleshooting
    9. Contact


================================================================================
1. OVERVIEW
================================================================================

AutoScreen is an end-to-end drug discovery workflow that streamlines the 
transition from target identification to lead optimization.

Pipeline Flow:
    
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │    Target    │───>│     Data     │───>│    Pre-      │
    │    Search    │    │  Acquisition │    │  processing  │
    └──────────────┘    └──────────────┘    └──────────────┘
                                                   │
           ┌───────────────────────────────────────┘
           │
           v
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │   Feature    │───>│    Model     │───>│     Lead     │
    │  Engineering │    │   Training   │    │ Optimization │
    └──────────────┘    └──────────────┘    └──────────────┘
                                                   │
                                                   v
                                            ┌──────────────┐
                                            │ Visualization│
                                            └──────────────┘

Key Capabilities:

    [*] Data Acquisition
        Direct querying of ChEMBL database for protein targets and 
        IC50 bioactivity data.
    
    [*] Preprocessing
        Automated SMILES cleaning, salt removal, and IC50 to pIC50 
        conversion.
    
    [*] Feature Engineering
        Simultaneous calculation of Lipinski descriptors and 2048-bit 
        ECFP4 Morgan fingerprints.
    
    [*] Model Benchmarking
        Comparison of 7 regression algorithms to identify the optimal 
        predictor for each target.
    
    [*] Lead Optimization
        Automated generation of molecular mutants using SELFIES to 
        enhance potency.
    
    [*] Visualization
        3D molecular embedding and GIF generation for structural 
        analysis of lead candidates.


================================================================================
2. PROJECT STRUCTURE
================================================================================

    autoscreen/
    │
    └── app.py .............. Complete application (single file)


================================================================================
3. KEY FEATURES
================================================================================

DATA PROCESSING
---------------
    • ChEMBL Client ........... Real-time API integration for protein 
                               target search
    • Bioactivity Filtering ... Automatic IC50 extraction with unit 
                               normalization
    • Molecular Standardization SMILES canonicalization via RDKit

MOLECULAR REPRESENTATIONS
-------------------------
    • Lipinski Rule of 5:
        - Molecular Weight (MolWt)
        - Partition Coefficient (LogP)
        - Hydrogen Bond Donors
        - Hydrogen Bond Acceptors

    • Fingerprints:
        - 2048-bit Morgan Fingerprints (ECFP4)
        - Radius: 2

MACHINE LEARNING MODELS
-----------------------
    Seven regression algorithms are automatically compared:

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │    1. Random Forest           5. Gradient Boosting             │
    │    2. Linear Regression       6. AdaBoost                      │
    │    3. Support Vector Regr.    7. K-Nearest Neighbors           │
    │    4. Decision Trees                                           │
    │                                                                 │
    │    >> Best model auto-selected based on lowest MSE <<          │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

LEAD OPTIMIZATION & VISUALIZATION
---------------------------------
    • SELFIES Mutation ........ Robust molecular string mutation 
                               ensuring chemical validity
    • 3D Engine ............... RDKit AllChem embedding with 
                               Matplotlib 3D projection
    • GIF Animation ........... Rotating view generation for 
                               structural inspection


================================================================================
4. INSTALLATION
================================================================================

REQUIREMENTS
------------
    Python Version: 3.9 - 3.11
    Package Manager: Conda (recommended)

STEP-BY-STEP INSTALLATION
-------------------------

    Step 1: Create Environment
    --------------------------
    $ conda create -n autoscreen python=3.10
    $ conda activate autoscreen


    Step 2: Install Dependencies
    ----------------------------
    $ pip install streamlit pandas chembl_webresource_client rdkit scikit-learn
    $ pip install numpy matplotlib seaborn joblib selfies pillow scipy


    Step 3: Run Application
    -----------------------
    $ streamlit run app.py


DEPENDENCIES LIST
-----------------
    +-----------------------------+----------------------------------+
    | Package                     | Purpose                          |
    +-----------------------------+----------------------------------+
    | streamlit                   | Web application framework        |
    | pandas                      | Data manipulation                |
    | chembl_webresource_client   | ChEMBL API access                |
    | rdkit                       | Molecular processing             |
    | scikit-learn                | Machine learning models          |
    | numpy                       | Numerical computations           |
    | matplotlib                  | Visualization & 3D plotting      |
    | seaborn                     | Statistical visualization        |
    | joblib                      | Model serialization              |
    | selfies                     | Molecular string mutations       |
    | pillow                      | Image processing for GIFs        |
    | scipy                       | Scientific computations          |
    +-----------------------------+----------------------------------+


================================================================================
5. USAGE GUIDE
================================================================================

STEP-BY-STEP WORKFLOW
---------------------

    STEP 1: Target Search
    .....................
    Enter a protein name (e.g., "EGFR") to retrieve available 
    ChEMBL target IDs.


    STEP 2: Training
    ................
    Select a target from the results. The application will:
        - Fetch bioactivity data from ChEMBL
        - Preprocess molecular structures
        - Calculate descriptors and fingerprints
        - Train and evaluate all 7 models


    STEP 3: Analysis
    ................
    Review the generated outputs:
        - Lipinski descriptor distributions
        - Model performance comparison table
        - Best model identification


    STEP 4: Screening
    .................
    The application automatically screens "world.csv" and saves 
    predictions to output files.


    STEP 5: Mutation
    ................
    Navigate to "Generating Drug Mutation" section to view:
        - Optimized molecular structures
        - 3D rotating animations
        - Predicted potency improvements


    STEP 6: Custom Upload (Optional)
    ................................
    Use the file uploader to predict potencies for your own 
    compound library.

    Required CSV format:
    +------------+------------------+
    | Column     | Description      |
    +------------+------------------+
    | smiles     | SMILES strings   |
    | zinc_id    | Compound IDs     |
    +------------+------------------+


================================================================================
6. CONFIGURATION
================================================================================

INTERNAL PARAMETERS
-------------------
The following parameters can be adjusted in app.py:

    +------------------+---------+----------------------------------------+
    | Parameter        | Default | Description                            |
    +------------------+---------+----------------------------------------+
    | nBits            | 2048    | ECFP4 fingerprint length               |
    | mutation_rate    | 3       | Mutations per SELFIES string           |
    | num_mutants      | 100     | Valid mutants generated per candidate  |
    | test_split       | 0.20    | Fraction of data for validation        |
    | num_frames       | 30      | Frames per GIF rotation                |
    +------------------+---------+----------------------------------------+


================================================================================
7. OUTPUT FILES & METRICS
================================================================================

PERFORMANCE METRICS
-------------------
The application generates a performance table containing:

    • MAE ......... Mean Absolute Error
    • MSE ......... Mean Squared Error  
    • R² .......... Coefficient of Determination

OPTIMIZATION RESULTS
--------------------
    File: best_mutants_table.csv
    
    Contents:
        - Original SMILES structures
        - Mutant SMILES with improved predicted pIC50
        - Predicted potency values

VISUAL ARTIFACTS
----------------
    Files: 1.gif, 2.gif, 3.gif, ...
    
    Description:
        Rotating 3D structure visualizations of:
        - Original lead compound
        - Optimized mutant counterparts


================================================================================
8. TROUBLESHOOTING
================================================================================

ISSUE: "Invalid SMILES" Warning
-------------------------------
    Cause:
        Input molecules contain non-standard or malformed SMILES strings.
    
    Solution:
        - Ensure input CSV uses standard canonical SMILES
        - The pipeline automatically skips unparseable molecules
        - Check RDKit documentation for SMILES formatting

................................................................................

ISSUE: ChEMBL API Timeout
-------------------------
    Cause:
        Database server is busy or network connection is unstable.
    
    Solution:
        - Wait a few minutes and restart the query
        - Check your internet connection
        - Try during off-peak hours for better response times

................................................................................

ISSUE: GIF Generation is Slow
-----------------------------
    Cause:
        3D embedding and frame rendering is CPU intensive.
    
    Solution:
        - Reduce 'num_frames' parameter (default: 30)
        - Reduce 'num_mutants' parameter (default: 100)
        - Consider running on a machine with more CPU cores


================================================================================
9. CONTACT
================================================================================

    Developed by: Biomini Team
    
    For issues, suggestions, or contributions, please contact the 
    development team.


================================================================================
                              END OF DOCUMENTATION
================================================================================
