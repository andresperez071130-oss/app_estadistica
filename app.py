import streamlit as st
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="App de Estadística - UPChiapas", layout="wide")

st.title("📊 Asistente de Análisis Estadístico e IA")
st.markdown("""
Esta aplicación permite explorar distribuciones y realizar pruebas de hipótesis 
con el apoyo de Inteligencia Artificial.
""")

# --- MÓDULO 1: CARGA DE DATOS ---
st.header("1. Carga de Datos")

# Opción para subir archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV aquí", type=["csv"])

if uploaded_file is not None:
    # Leer el dataset
    df = pd.read_csv(uploaded_file)
    
    # Mostrar vista previa de los datos
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    # Mostrar estadísticas descriptivas (EL REQUISITO DEL PASO 2)
    st.subheader("Resumen Estadístico (df.describe)")
    st.write(df.describe())
    
    # Mensaje de éxito para confirmar que todo va bien
    st.success("¡Datos cargados correctamente!")
else:
    st.info("Por favor, sube un archivo CSV para comenzar el análisis.")