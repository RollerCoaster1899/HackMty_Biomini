import streamlit as st
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
import seaborn as sns
import io
import warnings
from matplotlib.animation import FuncAnimation
import tempfile
from PIL import Image
import os
from rdkit.Chem import AllChem
import base64
from random import randint, choice
from rdkit.Chem import MolFromSmiles
from selfies import encoder, decoder, DecoderError


warnings.filterwarnings("ignore")

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
    st.image('biomini_gif.gif', width=200)

#----------------- APP NAME
# st.title('NOMBRE DEL PROYECTO')

# #------- EQUIPO QUE DESARROLLÓ EL PROYECTO
# st.subheader("""
# _App developed by Biomini team_
# """)


#----------------- APP NAME
st.markdown('<h1 style="color: black;">AutoScreen</h1>', unsafe_allow_html=True)
st.markdown('<h4 style="color: black;font-style: italic;">Accelerating Drug Discovery</h1>', unsafe_allow_html=True)

#------ EQUIPO QUE DESARROLLÓ EL PROYECTO
st.markdown('<h3 style="color: black; font-style: italic;">App developed by Biomini team</h2>', unsafe_allow_html=True)

#--------- AGREGANDO ESPACIO
st.text("")
#st.text("")


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
    # Filtrar y limpiar datos
    data_C1 = data[data.standard_value.notna()]
    data_C2 = data_C1[data_C1.canonical_smiles.notna()]
    data_C3 = data_C2.drop_duplicates(['canonical_smiles'])
    
    # Guardar datos limpios
    data_C3.to_csv(f'{selected_target}_S4_bioactivity_data.csv', index=False)
    
    # Calcular pIC50
    data_C3['pIC50'] = calculate_pIC50(data_C3)
    
    # Calcular descriptores de Lipinski
    lipinski_descriptors = data_C3['canonical_smiles'].apply(calculate_lipinski)
    lipinski_df = pd.DataFrame(lipinski_descriptors.tolist(), columns=['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors'])
    data_C3 = pd.concat([data_C3, lipinski_df], axis=1)
    
    # Calcular huellas dactilares ECFP4
    ecfp4_fps = data_C3['canonical_smiles'].apply(calculate_ecfp4)
    ecfp4_df = pd.DataFrame(ecfp4_fps.tolist(), columns=[f'ECFP4_{i}' for i in range(2048)])
    data_C3 = pd.concat([data_C3, ecfp4_df], axis=1)
    data_C3.to_csv(f'{selected_target}_S4_bioactivity_data.csv', index=False)
    
    return data_C3

# Función para clasificar la bioactividad basada en valores estándar
def Calculate_bioactivity_classes(selected_target, data_C3):
    selection = ["molecule_chembl_id", "canonical_smiles", "standard_value", "pIC50", "MolWt", "LogP", "NumHDonors", "NumHAcceptors"] + [f'ECFP4_{i}' for i in range(2048)]
    data_1 = data_C3[selection]
    data_1.to_csv(f'{selected_target}_S5_bioactivity_data.csv', index=False)
    return data_1

# Función para calcular pIC50 desde standard_value
def calculate_pIC50(df):
    pIC50 = []
    for value in df['standard_value']:
        try:
            molar_value = float(value) * 1e-9  # Convert nM to M
            if molar_value > 0:
                pIC50.append(-np.log10(molar_value))
            else:
                pIC50.append(np.nan)  # Handle non-positive values
        except (ValueError, TypeError):
            pIC50.append(np.nan)  # Handle invalid or missing values
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
        return list(fp.ToBitString())
    except Exception as e:
        print(f"Error processing SMILES for ECFP4: {smiles}, Error: {e}")
        return [None] * 2048

    
# Función para definir x y y para el modelo
def split_X_Y(df):
    # Ensure pIC50 is float
    df['pIC50'] = df['pIC50'].astype(float)
    df = df.dropna()

    # Define the target variable
    y = df['pIC50']
    # Define the feature variables
    feature_columns = [col for col in df.columns if col not in ['pIC50', 'molecule_chembl_id', 'canonical_smiles', 'standard_value']]
    X = df[feature_columns]
    # Convert features to float
    X = X.astype(float)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X, y, X_train, X_test, Y_train, Y_test

# Función para las gráficas de los datos de Lipinski
def plot_lipinski_distribution(selected_target, df):
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.histplot(df['MolWt'], kde=True, ax=axes[0, 0], color='blue')
    axes[0, 0].set_title('Distribution of MolWt')
    sns.histplot(df['LogP'], kde=True, ax=axes[0, 1], color='green')
    axes[0, 1].set_title('Distribution of LogP')
    sns.histplot(df['NumHDonors'], kde=True, ax=axes[1, 0], color='red')
    axes[1, 0].set_title('Distribution of NumHDonors')
    sns.histplot(df['NumHAcceptors'], kde=True, ax=axes[1, 1], color='purple')
    axes[1, 1].set_title('Distribution of NumHAcceptors')
    plt.tight_layout()
    
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    plt.close(fig)  
    return img_bytes

def evaluate_model(Y_test, Y_pred):
    metrics = {
        'MAE': mean_absolute_error(Y_test, Y_pred),
        'MSE': mean_squared_error(Y_test, Y_pred),
        'R^2': r2_score(Y_test, Y_pred)
    }
    return metrics

# Función para imprimir métricas
# def print_metrics(metrics):
#     st.write(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
#     st.write(f"Mean Squared Error (MSE): {metrics['MSE']:.4f}")
#     st.write(f"R^2 Score: {metrics['R^2']:.4f}")

def save_model(model, filename):
    """Save the trained model to a file."""
    joblib.dump(model, filename)
    print(f"The model has been saved to '{filename}'.")

# Función para entrenar y evaluar el modelo
def train_and_evaluate_model(model, model_name, X_train, X_test, Y_train, Y_test):
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    metrics = evaluate_model(Y_test, Y_pred)
    return metrics


# Crear un DataFrame para almacenar las métricas de todos los modelos
def create_metrics_table(model_metrics):
    # Extraer las métricas y los nombres de los modelos
    metrics_names = ['MAE', 'MSE', 'R^2']
    metrics_values = {name: [] for name in metrics_names}
    
    # Llenar el DataFrame con los valores correspondientes
    for model_name, metrics in model_metrics.items():
        for metric_name in metrics_names:
            metrics_values[metric_name].append(metrics[metric_name])
    
    # Crear el DataFrame
    metrics_df = pd.DataFrame(metrics_values, index=model_metrics.keys())
    return metrics_df

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
st.markdown('<p class="custom-text">Enter the protein name (e.g. EGFR):</p>', unsafe_allow_html=True)
user_input = st.text_area("", key='protein_target', placeholder='Enter protein target here', height=30)


#---- Evaluar si el usuario ha ingresado un valor
if user_input:
    targets = get_target_protein(user_input)
    
    if not targets.empty:
        # Seleccionar solo las columnas deseadas
        selected_columns = ['organism', 'target_chembl_id', 'pref_name', 'target_type']
        targets_filtered = targets[selected_columns]

        st.write("Targets retrived. Select target of interest:")
        st.dataframe(targets_filtered)

        # Crear una lista con los target_chembl_id para el selectbox
        target_names = targets["target_chembl_id"].tolist()

        # Seleccionar el target usando un selectbox
        selected_target_id = st.selectbox("Select a target based on its target ChEMBL identificator (target_chembl_id):", target_names)

        # Obtener el nombre del target seleccionado
        selected_target_name = targets.loc[targets['target_chembl_id'] == selected_target_id, 'pref_name'].values[0]

        # Mostrar el nombre del target seleccionado
        st.markdown(f"<h4 style='color: black; font-size: 14px;'>The selected target is (pref_name): {selected_target_name}</h4>", unsafe_allow_html=True)

        # Descargar los datos de bioactividad para el target seleccionado
        data = download_bioactivity_data(selected_target_id)
        
        if not data.empty:
            # Limpiar los datos de bioactividad
            data_C3 = Cleaning_bioactivity_data(selected_target_id, data)

            # Mostrar los datos de bioactividad limpios
            # st.write(f"Cleaned bioactivity data for {selected_target_name}:")
            # st.dataframe(data_C3[['activity_id', 'canonical_smiles', 'assay_chembl_id', 'standard_value', 'type', 'units']])
            
            # Clasificar los datos de bioactividad
            data_C5 = Calculate_bioactivity_classes(selected_target_id, data_C3)
            
            # Mostrar los datos de bioactividad clasificados en una tabla separada
            st.write(f"Bioactivity data with classification for {selected_target_name}:")
            st.dataframe(data_C5)

            st.text("")
            st.text("")


            st.markdown(f"""
                <div style="color: black; text-align: center;">
                    Distributions of Lipinski Descriptors of the molecules tested against {selected_target_name}
                </div>
            """, unsafe_allow_html=True)

            img_bytes = None
            # Llamar a la función para obtener la imagen
            img_bytes = plot_lipinski_distribution(selected_target_id, data_C3)

            # Mostrar la imagen en Streamlit
            st.image(img_bytes)

            # Aplicar la función split_X_Y a los datos procesados
            X, y, X_train, X_test, Y_train, Y_test = split_X_Y(data_C5)

            st.text("")
            st.text("")

            # Definir los modelos
            models = {
                'Random Forest': RandomForestRegressor(random_state=42),
                'Linear Regression': LinearRegression(),
                'Support Vector Regressor': SVR(),
                'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
                'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
                'AdaBoost Regressor': AdaBoostRegressor(random_state=42),
                'K-Nearest Neighbors Regressor': KNeighborsRegressor()
            }

            # Entrenar y evaluar cada modelo
            model_metrics = {}
            best_model = None
            best_model_name = None

            for name, model in models.items():
                metrics = train_and_evaluate_model(model, name, X_train, X_test, Y_train, Y_test)
                model_metrics[name] = metrics

                # Track the best model based on MSE
                if best_model is None or metrics['MSE'] < model_metrics[best_model_name]['MSE']:
                    best_model = model
                    best_model_name = name

            # Save the best model
            if best_model:
                save_model(best_model, 'best_model.pkl')
            
            # Mostrar las métricas obtenidas en cada modelo entrenado
            metrics_df = create_metrics_table(model_metrics)
            
            # Crear una tabla en HTML
            html_table = metrics_df.to_html(classes='table table-striped', index=False)

            # CSS para centrar la tabla
            centered_html = f"""
                <style>
                .table {{
                    width: 80%;
                    margin-left: auto;
                    margin-right: auto;
                }}
                .table th, .table td {{
                    padding: 10px;
                    text-align: center;
                }}
                .table th {{
                    background-color: #f2f2f2;
                }}
                </style>
                {html_table}
            """


            # Mostrar la tabla centrada en Streamlit
            st.write("Performance metrics of the tested models to predict the compound potency based on physicochemical and structural characteristics:")
            st.write(metrics_df.style.set_table_attributes('class="table table-striped"').set_table_styles(
                [{'selector': 'thead th', 'props': 'background-color: #f2f2f2; text-align: center;'}]
            ))

            # st.write("Performance metrics of the tested models to predict the compound potency based on physicochemical and structural characteristics:")
            # st.dataframe(metrics_df)

        else:
            st.write("No bioactivity data found for the selected target.")
    else:
        st.write("No targets found.")





@st.cache_data 
def load_model(filename):
    return joblib.load(filename)

# Función para limpiar y preprocesar datos
# def clean_smiles_df(predicting):
#     def calculate_lipinski(smiles):
#         try:
#             mol = Chem.MolFromSmiles(smiles)
#             if mol is None:
#                 raise ValueError(f"Invalid SMILES: {smiles}")
#             mol_wt = Descriptors.MolWt(mol)
#             log_p = Crippen.MolLogP(mol)
#             num_h_donors = Descriptors.NumHDonors(mol)
#             num_h_acceptors = Descriptors.NumHAcceptors(mol)
#             return mol_wt, log_p, num_h_donors, num_h_acceptors
#         except Exception as e:
#             print(f"Error processing SMILES: {smiles}, Error: {e}")
#             return None, None, None, None

#     def calculate_ecfp4(smiles):
#         try:
#             mol = Chem.MolFromSmiles(smiles)
#             if mol is None:
#                 raise ValueError(f"Invalid SMILES: {smiles}")
#             fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
#             return list(fp.ToBitString())
#         except Exception as e:
#             print(f"Error processing SMILES for ECFP4: {smiles}, Error: {e}")
#             return [None] * 2048

#     lipinski_data = []
#     ecfp4_data = []

#     for index, row in predicting.iterrows():
#         smiles = row['smiles']

#         if not isinstance(smiles, str):
#             print(f"Skipping invalid SMILES: {smiles}")
#             lipinski_data.append((None, None, None, None))
#             ecfp4_data.append([None] * 2048)
#             continue

#         lipinski_data.append(calculate_lipinski(smiles))
#         ecfp4_data.append(calculate_ecfp4(smiles))

#     lipinski_df = pd.DataFrame(lipinski_data, columns=['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors'])
#     ecfp4_df = pd.DataFrame(ecfp4_data, columns=[f'ECFP4_{i}' for i in range(2048)])
#     all_info = pd.concat([predicting, lipinski_df, ecfp4_df], axis=1)

#     feature_columns = [col for col in all_info.columns if col not in ['smiles', 'zinc_id']]
#     to_predict = all_info[feature_columns]
#     return to_predict

def process_data(predicting):
    if not isinstance(predicting, pd.DataFrame):
        raise TypeError("El objeto 'predicting' debe ser un DataFrame.")

    lipinski_data = []
    ecfp4_data = []

    for index, row in predicting.iterrows():
        smiles = row['smiles']

        if not isinstance(smiles, str):
            print(f"Skipping invalid SMILES: {smiles}")
            lipinski_data.append((None, None, None, None))
            ecfp4_data.append([None] * 2048)
            continue

        lipinski_result = calculate_lipinski(smiles)
        ecfp4_result = calculate_ecfp4(smiles)

        if lipinski_result is None or ecfp4_result is None:
            print(f"Error in calculations for SMILES: {smiles}")
            lipinski_data.append((None, None, None, None))
            ecfp4_data.append([None] * 2048)
        else:
            lipinski_data.append(lipinski_result)
            ecfp4_data.append(ecfp4_result)

    lipinski_df = pd.DataFrame(lipinski_data, columns=['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors'])
    ecfp4_df = pd.DataFrame(ecfp4_data, columns=[f'ECFP4_{i}' for i in range(2048)])
    all_info = pd.concat([predicting, lipinski_df, ecfp4_df], axis=1)

    feature_columns = [col for col in all_info.columns if col not in ['smiles', 'zinc_id']]
    to_predict = all_info[feature_columns]
    return to_predict

# Cargar modelos
model_filename = 'best_model.pkl'
model = load_model(model_filename)


# Define la ruta del archivo CSV
file_path = "world.csv"

predicting = pd.read_csv(file_path)

# Limpia el DataFrame si es necesario
to_predict = process_data(predicting)

# Realizar predicciones
if not to_predict.empty:
    predictions = model.predict(to_predict)
    predicting['predictions'] = predictions

    predicting = predicting.drop('zinc_id', axis=1)

    # Create a DataFrame with SMILES and predicted pIC50 values
    df = pd.DataFrame({
    'pIC50': predictions,
    'SMILES': predicting["smiles"]
    })

    # Sort the DataFrame by pIC50 in descending order
    df_sorted = df.sort_values(by='pIC50', ascending=False)

    # Mostrar predicciones
    st.write("Predictions:")
    st.dataframe(predicting)

    st.text("")

    # Mostrar gráficos si es necesario
    st.markdown(f"""
                <div style="color: black; text-align: center; font-weight: bold">
                    Potency Distribution:
                </div>
            """, unsafe_allow_html=True)
    fig, ax = plt.subplots()
    sns.histplot(predictions, kde=True, ax=ax)
    st.pyplot(fig)


    # Guardar las predicciones en un archivo CSV
    csv = predicting.to_csv(index=False).encode('utf-8')

    # Codificar el CSV en base64
    csv_base64 = base64.b64encode(csv).decode()

    # Crear el botón con estilos personalizados usando HTML
    button_html = f"""
        <style>
        .download-button {{
            background-color: black;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            font-weight: bold;
            border: none;
            cursor: pointer;
        }}
        .download-button:hover {{
            background-color: #333;
        }}
        </style>
        <a class="download-button" href="data:text/csv;base64,{csv_base64}" download="predictions.csv">Download Predictions</a>
    """

    # Mostrar el botón estilizado en Streamlit
    st.markdown(button_html, unsafe_allow_html=True)



def generate_3d_molecule_image(mol, rotation_angle):
    """Generate a 3D image of a molecule rotated by rotation_angle degrees."""
    # Create a 3D plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Get molecule coordinates
    conf = mol.GetConformer()
    coords = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    coords = np.array(coords)

    # Rotate coordinates
    theta = np.radians(rotation_angle)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    rotated_coords = np.dot(coords, rotation_matrix.T)

    # Draw the molecule
    ax.scatter(rotated_coords[:, 0], rotated_coords[:, 1], rotated_coords[:, 2])
    for bond in mol.GetBonds():
        start = rotated_coords[bond.GetBeginAtomIdx()]
        end = rotated_coords[bond.GetEndAtomIdx()]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'k-')

    # Save the figure to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(temp_file.name)
    plt.close(fig)
    return temp_file.name

def create_gif(mol, num_frames=30, filename='molecule.gif', frame_duration=200):
    """Create a rotating 3D GIF of the molecule."""
    images = []
    for i in range(num_frames):
        angle = (i / num_frames) * 360
        img_path = generate_3d_molecule_image(mol, angle)
        
        # Open the image with 'with' to ensure it is closed properly
        with Image.open(img_path) as img:
            images.append(img.copy())  # Copy the image to avoid issues with closing

        # Clean up the temporary file
        os.remove(img_path)

    # Create GIF with adjusted frame duration
    images[0].save(filename, save_all=True, append_images=images[1:], duration=frame_duration, loop=0)
    return filename



# Añadir una sección para generar imágenes 3D de moléculas y crear GIFs
# st.markdown("<h3 style='color: black;'>Generar GIF 3D de una molécula</h3>", unsafe_allow_html=True)

smiles_list = df_sorted['SMILES'].head(3)

# # Iterar sobre los primeros 3 SMILES
# for i, smiles_input in enumerate(smiles_list):
#     st.write(f"Procesando molécula {i+1}: {smiles_input}")
    
#     mol = Chem.MolFromSmiles(smiles_input)  # Convertir SMILES a molécula
#     if mol:
#         # Generar coordenadas 3D de la molécula
#         AllChem.EmbedMolecule(mol)

#         # Crear el GIF (suponiendo que tienes la función create_gif)
#         gif_filename = create_gif(mol, num_frames=30, frame_duration=200)

#         # Mostrar el GIF en el centro
#         left_co, cent_co, last_co = st.columns(3)
#         with cent_co:
#             st.image(gif_filename, caption=f"GIF de la molécula {i+1} 3D")
#     else:
#         st.error(f"El SMILES de la molécula {i+1} no es válido.")



def IsCorrectSMILES(smiles):
    if len(smiles) == 0:
        return False
    try:
        resMol = MolFromSmiles(smiles, sanitize=True)
        return resMol is not None
    except Exception:
        return False

def tokenize_selfies(selfies):
    tokens = []
    stack = []
    start = 0
    for i, char in enumerate(selfies):
        if char == '[':
            if stack:
                tokens.append(selfies[start:i])
            start = i
            stack.append('[')
        elif char == ']':
            stack.pop()
            if not stack:
                tokens.append(selfies[start:i+1])
                start = i+1
    if start < len(selfies):
        tokens.append(selfies[start:])
    return tokens

def detokenize_selfies(tokens):
    return ''.join(tokens)

def mutate_selfies(selfies, num_mutations, smiles_symbols):
    tokens = tokenize_selfies(selfies)
    if len(tokens) <= 1:
        return None

    for _ in range(num_mutations):
        mol_idx = randint(0, len(tokens) - 1)
        symbol = choice(smiles_symbols)
        tokens[mol_idx] = symbol

    mutated_selfies = detokenize_selfies(tokens)
    return mutated_selfies

def mutate_smiles(smiles, num_mutations, smiles_symbols):
    selfies = encoder(smiles)
    mutated_selfies = mutate_selfies(selfies, num_mutations, smiles_symbols)

    if mutated_selfies:
        try:
            mutated_smiles = decoder(mutated_selfies)
            if IsCorrectSMILES(mutated_smiles):
                return mutated_smiles
        except DecoderError:
            pass  # Handle or log the decoder error if needed
    return None

def generate_valid_mutants(smiles, num_mutants, smiles_symbols, max_attempts=1000):
    mutants = set()
    attempts = 0

    while len(mutants) < num_mutants and attempts < max_attempts:
        attempts += 1
        mutated_smiles = mutate_smiles(smiles, num_mutations=3, smiles_symbols=smiles_symbols)
        if mutated_smiles and mutated_smiles not in mutants:
            mutants.add(mutated_smiles)

    return list(mutants)

# def clean_smiles(smiles):
#     """Process a single SMILES string to return its feature vector."""
#     def calculate_lipinski(mol):
#         """Calculate Lipinski descriptors."""
#         mol_wt = Descriptors.MolWt(mol)
#         log_p = Crippen.MolLogP(mol)
#         num_h_donors = Descriptors.NumHDonors(mol)
#         num_h_acceptors = Descriptors.NumHAcceptors(mol)
#         return mol_wt, log_p, num_h_donors, num_h_acceptors

def clean_smiles(smiles):
    """Process a single SMILES string to return its feature vector."""
    def calculate_lipinski(mol):
        """Calculate Lipinski descriptors."""
        mol_wt = Descriptors.MolWt(mol)
        log_p = Crippen.MolLogP(mol)
        num_h_donors = Descriptors.NumHDonors(mol)
        num_h_acceptors = Descriptors.NumHAcceptors(mol)
        return mol_wt, log_p, num_h_donors, num_h_acceptors

    def calculate_ecfp4(mol):
        """Calculate ECFP4 fingerprint."""
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
        return list(fp.ToBitString())

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Calculate Lipinski descriptors
        lipinski = calculate_lipinski(mol)

        # Calculate ECFP4 descriptors
        ecfp4 = calculate_ecfp4(mol)

        # Combine all features into a single vector
        features = list(lipinski) + ecfp4

        return pd.DataFrame([features], columns=[
            'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors'] + [f'ECFP4_{i}' for i in range(2048)]
        )

    except Exception as e:
        print(f"Error processing SMILES: {smiles}, Error: {e}")
        # Return a DataFrame with NaNs for features if an error occurs
        return pd.DataFrame([np.nan] * (4 + 2048), columns=[
            'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors'] + [f'ECFP4_{i}' for i in range(2048)]
        )


# def predict_single_molecule(smiles, model):
#     """Predict the pIC50 value for a single molecule."""
#     to_predict = clean_smiles(smiles)
#     if to_predict.isnull().values.any():
#         print("Prediction cannot be made due to invalid input features.")
#         return None

def predict_single_molecule(smiles, model):
     """Predict the pIC50 value for a single molecule."""
     to_predict = clean_smiles(smiles)
     if to_predict is None:
         print(f"Cleaning failed for SMILES: {smiles}")
         return None
     if to_predict.isnull().values.any():
         print("Prediction cannot be made due to invalid input features.")
         return None
     prediction = model.predict(to_predict)
     return prediction[0]

# def predict_single_molecule(smiles, model):
#     """Predict the pIC50 value for a single molecule."""
#     to_predict = clean_smiles(smiles)
#     if to_predict is None:
#         print(f"Cleaning failed for SMILES: {smiles}")
#         return None
#     if to_predict.isnull().values.any():
#         print("Prediction cannot be made due to invalid input features.")
#         return None

#     prediction = model.predict(to_predict)  # Asegúrate de que el modelo esté definido y funcionando
#     print(f"Prediction for {smiles}: {prediction}")
#     return prediction

model = joblib.load('best_model.pkl')


# Initialize dictionary to hold the best mutants per original molecule
best_mutants = {}

# List to collect data for the table
table_data = []

# Process each SMILES string
for smiles in smiles_list:
    # Get pIC50 of the original SMILES
    original_pIC50 = predict_single_molecule(smiles, model)

    # Generate mutants
    num_desired_mutants = 100
    smiles_symbols = 'FONC()=#12345'  # Define this if necessary
    mutants = generate_valid_mutants(smiles, num_desired_mutants, smiles_symbols)

    # Evaluate mutants
    mutant_scores = []
    for mutant in mutants:
        prediction = predict_single_molecule(mutant, model)
        if prediction is not None:
            mutant_scores.append((mutant, prediction))
        else:
            # Handle mutants with invalid predictions
            mutant_scores.append((mutant, -float('inf')))  # Use a very low value for sorting
    
    # Sort mutants by predicted pIC50 value in descending order
    mutant_scores.sort(key=lambda x: x[1], reverse=True)
    top_mutants = [mutant for mutant, _ in mutant_scores[:2]]

    # Store the best mutants for this SMILES
    best_mutants[smiles] = top_mutants

    # Collect data for the table
    if len(top_mutants) > 0:
        pIC50_1 = predict_single_molecule(top_mutants[0], model)
        if len(top_mutants) > 1:
            pIC50_2 = predict_single_molecule(top_mutants[1], model)
        else:
            pIC50_2 = 'N/A'
        table_data.append([smiles, original_pIC50, top_mutants[0], pIC50_1, top_mutants[1] if len(top_mutants) > 1 else 'N/A', pIC50_2])

# Create a DataFrame and save it as a CSV file
df = pd.DataFrame(table_data, columns=['Original SMILES', 'Original pIC50', 'Best Mutant 1 SMILES', 'pIC50 (Best Mutant 1)', 'Best Mutant 2 SMILES', 'pIC50 (Best Mutant 2)'])
df.to_csv('best_mutants_table.csv', index=False)

# Initialize counters for GIF naming
gif_counter = 1

# Generate 3D visualizations for the best mutants
for original_smiles, mutants in best_mutants.items():
    # Process the original molecule
    mol_original = Chem.MolFromSmiles(original_smiles)
    if mol_original:
        AllChem.EmbedMolecule(mol_original)  # Generate 3D coordinates
        filename = f'{gif_counter}.gif'
        print(f"Saving GIF for original molecule: {filename}")
        create_gif(mol_original, num_frames=30, filename=filename, frame_duration=200)
        gif_counter += 1
    else:
        print(f"Invalid SMILES string for original molecule: {original_smiles}")

    # Process top mutants
    for i, smiles in enumerate(mutants):
        mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to molecule
        if mol:
            AllChem.EmbedMolecule(mol)  # Generate 3D coordinates
            filename = f'{gif_counter}.gif'
            print(f"Saving GIF for mutant {i+1}: {filename}")
            create_gif(mol, num_frames=30, filename=filename, frame_duration=200)
            gif_counter += 1
        else:
            print(f"Invalid SMILES string for mutant {i+1}: {smiles}")

## Initialize counters for GIF naming
gif_counter = 1



# Display GIFs and data using Streamlit
st.title("Visualization of Molecules and Mutants")

# Counter to keep track of the GIF files
# gif_counter = 1


best_mutants_df = pd.read_csv('best_mutants_table.csv')

# Function to display a single row of GIFs with column titles
def display_row(original_smiles, mutants, start_gif_counter):
    # Create columns for the row
    col1, col2, col3 = st.columns(3)

    # Add titles to the columns
    with col1:
        st.write("Original")
    
    with col2:
        st.write("Mutant 1")
    
    with col3:
        st.write("Mutant 2")


    #best_mutants_df = 
    # Fetch the data from the dataframe
    row = best_mutants_df[best_mutants_df['Original SMILES'] == original_smiles].iloc[0]
    
    # Display original molecule GIF and its details
    with col1:
        st.image(f'{start_gif_counter}.gif')
        st.write(f"SMILES Code: {original_smiles}")
        st.write(f"pIC50: {row['Original pIC50']}")
    
    mutant_gif_counter = start_gif_counter + 1

    # Display mutant 1 GIF and its details
    with col2:
        st.image(f'{mutant_gif_counter}.gif')
        st.write(f"SMILES Code: {mutants[0]}")
        st.write(f"pIC50: {row['pIC50 (Best Mutant 1)']}")
    
    mutant_gif_counter += 1
    # Display mutant 2 GIF and its details
    with col3:
        if len(mutants) > 1:
            st.image(f'{mutant_gif_counter}.gif')
            st.write(f"SMILES Code: {mutants[1]}")
            st.write(f"pIC50: {row['pIC50 (Best Mutant 2)']}")
        else:
            st.write("No data")

# Initialize starting GIF counter
gif_counter = 1

# Iterate through each original molecule and its mutants
for original_smiles, mutants in best_mutants.items():
    display_row(original_smiles, mutants, start_gif_counter=gif_counter)
    # Update the GIF counter for the next row
    gif_counter += 3
    st.write("---")