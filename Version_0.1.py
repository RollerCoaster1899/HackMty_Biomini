import streamlit as st
import pandas as pd
from chembl_webresource_client.new_client import new_client

#----------------- APP NAME
st.title('NOMBRE DEL PROYECTO')


# page_bg_style = """
# <style>
# [data-testid="stAppViewContainer"] {
#     background-color: #fdf5e6;
# }
# </style>
# """
# st.markdown(page_bg_style, unsafe_allow_html=True)

#------- EQUIPO QUE DESARROLLÓ EL PROYECTO
st.subheader("""
_App developed by Biomini team_
""")

#--------- AGREGANDO ESPACIO
st.text("")
st.text("")
st.text("")


#--------- AGREGAR DESCRIPCIÓN DE LA APP
st.markdown('<p style="font-size:24px;">APP DESCRIPTION</p>', unsafe_allow_html=True)


#--------- AGREGANDO ESPACIO
st.text("")

#--------- DEFINIR LOS CLIENTES (PERSONA QUE SUBMITE EL REQUEST)
class NewClient:
    def __init__(self):
        pass
    
    class Target:
        def search(self, target_protein):
            return [{"name": "Protein A", "id": "CHEMBL1234"}, {"name": "Protein B", "id": "CHEMBL5678"}]
    
    class Activity:
        def filter(self, target_chembl_id, standard_type):
            # Simulación de los resultados de la actividad
            if target_chembl_id == "CHEMBL1234":
                return [
                    {"activity_id": 1, "canonical_smiles": "CCO", "assay_chembl_id": "CHEMBL1", "value": 50, "type": "IC50", "units": "nM"},
                    {"activity_id": 2, "canonical_smiles": "CCN", "assay_chembl_id": "CHEMBL2", "value": 100, "type": "IC50", "units": "nM"}
                ]
            elif target_chembl_id == "CHEMBL5678":
                return [
                    {"activity_id": 3, "canonical_smiles": "CCC", "assay_chembl_id": "CHEMBL3", "value": 150, "type": "IC50", "units": "nM"}
                ]
            return []
    
    target = Target()
    activity = Activity()

new_client = NewClient()

#------- INGRESAR EL NOMBRE DE LA PROTEÍNA TARGET
user_input = st.text_area("Put here your protein target name")

# Función para buscar el target de la proteína
def get_target_protein(target_protein):
    target = new_client.target
    target_query = target.search(target_protein) 
    targets = pd.DataFrame.from_dict(target_query)
    return targets

# Función para descargar los datos de bioactividad
def Download_bioactivity_data(selected_target):
    activity = new_client.activity
    res = activity.filter(target_chembl_id=selected_target, standard_type="IC50")
    data = pd.DataFrame.from_dict(res)
    if not data.empty:
        data.to_csv(f'{selected_target}_NC_bioactivity_data.csv', index=False)
    return data

#---- Evaluar si el usuario ha ingresado un valor
if user_input:
    targets = get_target_protein(user_input)
    
    if not targets.empty:
        st.write("Targets found:")
        st.dataframe(targets)

        # Crear una lista con los nombres de los targets para el selectbox
        target_names = targets["name"].tolist()

        # Seleccionar el target usando un selectbox
        selected_target_name = st.selectbox("Select a target:", target_names)

        # Obtener el ID del target seleccionado
        selected_target_id = targets.loc[targets['name'] == selected_target_name, 'id'].values[0]

        # Mostrar el nombre del target seleccionado
        st.markdown(f"<h3 style='font-size:15px;'>The selected target is: {selected_target_name}</h3>", unsafe_allow_html=True)

        # Descargar y mostrar los datos de bioactividad
        data = Download_bioactivity_data(selected_target_id)
        if not data.empty:
            st.write("Bioactivity data:")
            data_shown = data[['activity_id', 'canonical_smiles', 'assay_chembl_id', 'value', 'type', 'units']]
            st.dataframe(data_shown)
        else:
            st.write("No bioactivity data found for the selected target.")
    else:
        st.write("No targets found.")
