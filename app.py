import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from config import configurar_ia  # Importamos tu función segura

# Configuración estética
st.set_page_config(page_title="Estadística Pro - UPChiapas", layout="wide")

# Estilo personalizado con CSS para mejorar los títulos
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Plataforma de Análisis Estadístico e Inferencia")

# --- SECCIÓN 1: ADQUISICIÓN DE DATOS (Siempre visible) ---
with st.sidebar:
    st.header("Configuración de Datos")
    opcion_datos = st.radio("Fuente de datos:", ["Subir CSV", "Datos Sintéticos"])
    
    df = None
    if opcion_datos == "Subir CSV":
        uploaded_file = st.file_uploader("Carga tu archivo", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
    else:
        n_puntos = st.number_input("n:", min_value=30, value=100)
        media_s = st.number_input("Media:", value=50.0)
        desv_s = st.number_input("Desviación:", value=10.0)
        if st.button("Generar Dataset"):
            df = pd.DataFrame({'Variable_Sintetica': np.random.normal(media_s, desv_s, n_puntos)})

# --- SECCIÓN LÓGICA: SOLO SI HAY DATOS ---
if df is not None:
    # Creamos pestañas para organizar la interfaz
    tab1, tab2, tab3 = st.tabs(["📈 Visualización", "🧪 Prueba de Hipótesis", "🤖 Asistente IA"])

    # --- TAB 1: VISUALIZACIÓN ---
    with tab1:
        st.header("Análisis Exploratorio")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        variable = st.selectbox("Variable a analizar:", numeric_columns)
        data = df[variable].dropna()

        col1, col2, col3 = st.columns(3)
        with col1:
            fig, ax = plt.subplots(); sns.histplot(data, color="skyblue", ax=ax); st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(); sns.kdeplot(data, fill=True, color="green", ax=ax); st.pyplot(fig)
        with col3:
            fig, ax = plt.subplots(); sns.boxplot(y=data, color="tomato", ax=ax); st.pyplot(fig)

    # --- TAB 2: PRUEBA Z ---
    with tab2:
        st.header("Prueba Z de una Muestra")
        c1, c2 = st.columns(2)
        with c1:
            mu_0 = st.number_input("H0 (μ):", value=float(data.mean()))
            tipo = st.selectbox("H1:", ["Bilateral (≠)", "Superior (>)", "Inferior (<)"])
        with c2:
            alpha = st.select_slider("Significancia (α):", options=[0.01, 0.05, 0.10], value=0.05)
            sigma = st.number_input("Sigma (σ):", value=float(data.std()))

        # Lógica de cálculo (omito detalles por brevedad, usa la que ya tenemos)
        z_stat = (data.mean() - mu_0) / (sigma / np.sqrt(len(data)))
        # ... (Cálculo de p_val y rechazo aquí) ...

        st.metric("Resultado", "Rechazar H0" if z_stat > 1.96 else "No Rechazar H0") # Ejemplo simple

    # --- TAB 3: ASISTENTE IA (Solo aparece cuando hay datos cargados) ---
    with tab3:
        st.header("Consultoría con IA")
        st.info("Esta sección utiliza Gemini para interpretar los resultados de la pestaña anterior.")
        
        if st.button("🪄 Generar Análisis Experto"):
            model = configurar_ia()
            if model:
                with st.spinner("Analizando..."):
                    # Aquí va tu prompt con los datos de tab2
                    prompt = f"Explica una prueba Z con z={z_stat:.2f} para la variable {variable}."
                    response = model.generate_content(prompt)
                    st.markdown("### Interpretación de la IA")
                    st.write(response.text)
            else:
                st.error("Configura tu API Key en el archivo .env")

else:
    # Mensaje inicial cuando la app está vacía
    st.image("https://cdn-icons-png.flaticon.com/512/4090/4090434.png", width=100)
    st.title("Bienvenido")
    st.info("Por favor, usa el menú lateral para cargar un archivo CSV o generar datos sintéticos y comenzar el análisis.")