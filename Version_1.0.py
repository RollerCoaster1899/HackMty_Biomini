import streamlit as st
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
import numpy as np

#----------------- CAMBIAR EL COLOR DE FONDO
st.markdown(
    """
    <style>
    .main {
        background-color: white;
        color: black;
    }
    h1, h2, h3, h4, h5, h6, p, span {
        color: black !important; /* Aplicar color negro a todos los encabezados y párrafos */
    }
    </style>
    """,
    unsafe_allow_html=True
)


#----------------- Agregar logotipo del equipo
left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image("Logo.jpeg", width=200)


#----------------- APP NAME
# st.title('NOMBRE DEL PROYECTO')

# #------- EQUIPO QUE DESARROLLÓ EL PROYECTO
# st.subheader("""
# _App developed by Biomini team_
# """)


#----------------- APP NAME
st.markdown('<h1 style="color: black;">NOMBRE DEL PROYECTO</h1>', unsafe_allow_html=True)

#------- EQUIPO QUE DESARROLLÓ EL PROYECTO
st.markdown('<h3 style="color: black; font-style: italic;">App developed by Biomini team</h2>', unsafe_allow_html=True)

#--------- AGREGANDO ESPACIO
st.text("")
st.text("")

#------------ Agregar descripción de la aplicación
st.markdown(
    """
    <style>
    .main {
        background-color: white; 
    }
    .justified-text {
        text-align: justify; /* Justifica el texto */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('''
<p class="justified-text" style="font-size:18px; color: black;">
This application is a powerful tool for bioactivity analysis and therapeutic molecule discovery. It offers:
</p>

<ul class="justified-text" style="font-size:16px; color: black;">
    <li><b>Protein Target Search</b>: Enter the name of a target protein to retrieve relevant data from the ChEMBL database.</li>
    <li><b>Bioactivity Data Management</b>: Download and clean bioactivity data associated with the selected target. The application calculates essential metrics, molecular descriptors, and processes the data for further analysis.</li>
    <li><b>Bioactivity Classification</b>: Classify compounds based on their activity levels into categories such as "active," "intermediate," or "inactive" to streamline your analysis.</li>
    <li><b>Feature Extraction</b>: Generate and save feature datasets, including pIC50 values and molecular descriptors, for use in machine learning models.</li>
    <li><b>Machine Learning Modeling</b>: Build and apply machine learning models to predict which therapeutic molecules are most likely to interact effectively with the target protein. This includes training models to identify potential drug candidates based on the processed bioactivity data.</li>
</ul>

<p class="justified-text" style="font-size:16px; color: black;">
This tool is designed to support data-driven drug discovery by integrating bioactivity data processing with advanced machine learning techniques, helping researchers identify promising therapeutic molecules for further development.
</p>
''', unsafe_allow_html=True)

# Función para buscar el target de la proteína utilizando la API de ChEMBL
def get_target_protein(target_protein):
    target = new_client.target
    target_query = target.search(target_protein) 
    targets = pd.DataFrame.from_dict(target_query)
    return targets

# Función para descargar los datos de bioactividad utilizando la API de ChEMBL
def download_bioactivity_data(selected_target):
    activity = new_client.activity
    res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")
    data = pd.DataFrame.from_dict(res)
    data.to_csv(f'{selected_target}_S3_bioactivity_data.csv', index=False)
    return data

# Función para limpiar los datos de bioactividad y calcular características adicionales
def Cleaning_bioactivity_data(selected_target, data):
    data_C1 = data[data.standard_value.notna()]
    data_C2 = data_C1[data_C1.canonical_smiles.notna()]
    data_C3 = data_C2.drop_duplicates(['canonical_smiles'])
    data.to_csv(f'{selected_target}_S4_bioactivity_data.csv', index=False)
    # Calcular pIC50
    data_C3['pIC50'] = calculate_pIC50(data_C3)
    
    # Calcular Lipinski descriptors
    lipinski_descriptors = data_C3['canonical_smiles'].apply(calculate_lipinski)
    lipinski_df = pd.DataFrame(lipinski_descriptors.tolist(), columns=['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors'])
    data_C3 = pd.concat([data_C3, lipinski_df], axis=1)
    
    # Calcular ECFP4 fingerprints
    ecfp4_fps = data_C3['canonical_smiles'].apply(calculate_ecfp4)
    ecfp4_df = pd.DataFrame(ecfp4_fps.tolist(), columns=[f'ECFP4_{i}' for i in range(2048)])
    data_C3 = pd.concat([data_C3, ecfp4_df], axis=1)
    
    data_C3.to_csv(f'{selected_target}_S4_bioactivity_data.csv', index=False)
    return data_C3

# Función para clasificar la bioactividad basada en valores estándar
def Calculate_bioactivity_classes(selected_target, data_C3):
    bioactivity_class = []

    for i in data_C3.standard_value:
        if float(i) >= 10000:
            bioactivity_class.append("inactive")
        elif float(i) <= 1000:
            bioactivity_class.append("active")
        else:
            bioactivity_class.append("intermediate")

    bioactivity_class = pd.Series(bioactivity_class, name='bioactivity_class')
    data_C4 = pd.concat([data_C3, bioactivity_class], axis=1)
    selection = ["molecule_chembl_id", "canonical_smiles", "standard_value", "bioactivity_class", "pIC50", "MolWt", "LogP", "NumHDonors", "NumHAcceptors"] + [f'ECFP4_{i}' for i in range(2048)]
    data_C5 = data_C4[selection]
    data_C5.to_csv(f'{selected_target}_S5_bioactivity_data.csv', index=False)

    return bioactivity_class, data_C5

# Función para calcular pIC50 desde standard_value
def calculate_pIC50(df):
    pIC50 = []
    for value in df['standard_value']:
        molar_value = float(value) * 1e-9  # Convert nM to M
        if molar_value > 0:
            pIC50.append(-np.log10(molar_value))
        else:
            pIC50.append(None)  # Avoid math errors
    return pIC50

# Función para calcular descriptores de Lipinski
def calculate_lipinski(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Calcular descriptores de Lipinski
        mol_wt = Descriptors.MolWt(mol)
        log_p = Crippen.MolLogP(mol)
        num_h_donors = Descriptors.NumHDonors(mol)
        num_h_acceptors = Descriptors.NumHAcceptors(mol)
        return mol_wt, log_p, num_h_donors, num_h_acceptors

    except Exception as e:
        print(f"Error processing SMILES: {smiles}, Error: {e}")
        return None, None, None, None


# Función para calcular huellas dactilares ECFP4 usando MorganGenerator
def calculate_ecfp4(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Calcular huella dactilar ECFP4 (2048 bits)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
        # Convertir huella dactilar a una lista de 0s y 1s
        return list(fp.ToBitString())

    except Exception as e:
        print(f"Error processing SMILES for ECFP4: {smiles}, Error: {e}")
        return [None] * 2048
    
# Función para definir x y y para el modelo
def split_X_Y(df):
    # Ensure pIC50 is float
    df['pIC50'] = df['pIC50'].astype(float)

    # Define the target variable
    Y = df['pIC50']
    Y.to_csv(f'{selected_target_id}_Y_data.csv', index=False)

    # Define the feature variables
    feature_columns = [col for col in df.columns if col not in ['pIC50', 'molecule_chembl_id', 'canonical_smiles', 'standard_value', 'bioactivity_class']]
    X = df[feature_columns]

    # Convert features to float
    X = X.astype(float)
    X.to_csv(f'{selected_target_id}_X_data.csv', index=False)

    return X, Y

# Aplicar CSS para eliminar el espacio entre el texto y el área de texto
st.markdown(
    """
    <style>
    .custom-text {
        margin-bottom: 0px; /* Ajustar el margen inferior del texto */
        padding-bottom: 0px; /* Ajustar el padding inferior del texto */
        font-size: 16px; /* Opcional: Ajustar el tamaño de la fuente si es necesario */
    }
    .custom-textarea {
        margin-top: 0px; /* Ajustar el margen superior del área de texto */
        padding-top: 0px; /* Ajustar el padding superior del área de texto */
        height: 30px; /* Ajustar la altura del área de texto si es necesario */
    }
    .stTextArea {
        margin-top: 0px; /* Ajustar el margen superior del área de texto */
    }
    </style>
    """,
    unsafe_allow_html=True
)

#------- INGRESAR EL NOMBRE DE LA PROTEÍNA TARGET
st.markdown('<p class="custom-text">Put here your protein target name</p>', unsafe_allow_html=True)
user_input = st.text_area("", key='protein_target', placeholder='Enter protein target here', height=30)


#---- Evaluar si el usuario ha ingresado un valor
if user_input:
    targets = get_target_protein(user_input)
    
    if not targets.empty:
        # Seleccionar solo las columnas deseadas
        selected_columns = ['organism', 'pref_name', 'target_chembl_id', 'target_type']
        targets_filtered = targets[selected_columns]

        st.write("Targets found:")
        st.dataframe(targets_filtered)

        # Crear una lista con los target_chembl_id para el selectbox
        target_names = targets["target_chembl_id"].tolist()

        # Seleccionar el target usando un selectbox
        selected_target_id = st.selectbox("Select a target:", target_names)

        # Obtener el nombre del target seleccionado
        selected_target_name = targets.loc[targets['target_chembl_id'] == selected_target_id, 'pref_name'].values[0]

        # Mostrar el nombre del target seleccionado
        st.markdown(f"<h3 style='color: black;'>The selected target is: {selected_target_name}</h3>", unsafe_allow_html=True)

        # Descargar los datos de bioactividad para el target seleccionado
        data = download_bioactivity_data(selected_target_id)
        
        if not data.empty:
            # Limpiar los datos de bioactividad
            data_C3 = Cleaning_bioactivity_data(selected_target_id, data)

            # Mostrar los datos de bioactividad limpios
            st.write(f"Cleaned bioactivity data for {selected_target_name}:")
            st.dataframe(data_C3[['activity_id', 'canonical_smiles', 'assay_chembl_id', 'standard_value', 'type', 'units']])
            
            # Clasificar los datos de bioactividad
            bioactivity_class, data_C5 = Calculate_bioactivity_classes(selected_target_id, data_C3)
            
            # Mostrar los datos de bioactividad clasificados en una tabla separada
            st.write(f"Bioactivity data with classification for {selected_target_name}:")
            st.dataframe(data_C5)

            # Aplicar la función split_X_Y a los datos procesados
            X, Y = split_X_Y(data_C5)
        else:
            st.write("No bioactivity data found for the selected target.")
    else:
        st.write("No targets found.")

        
