import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuración de la página
st.set_page_config(page_title="App de Estadística - UPChiapas", layout="wide")

st.title("📊 Asistente de Análisis Estadístico e IA")
st.markdown("Desarrollado para la documentación del proceso creativo y uso de IA Generativa.")

# --- MÓDULO 1: CARGA DE DATOS ---
st.header("1. Carga de Datos")
uploaded_file = st.file_uploader("Sube tu archivo CSV aquí", type=["csv"])

if uploaded_file is not None:
    # Leer el dataset
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Vista previa y Resumen")
    col_pre1, col_pre2 = st.columns(2)
    
    with col_pre1:
        st.write("Primeros 5 registros:")
        st.dataframe(df.head())
    
    with col_pre2:
        st.write("Estadísticas descriptivas:")
        st.write(df.describe())
    
    # --- MÓDULO 2: VISUALIZACIÓN DE DISTRIBUCIONES ---
    st.divider()
    st.header("2. Visualización de Distribuciones")

    # Selección de variables numéricas
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if numeric_columns:
        variable = st.selectbox("Selecciona una variable para analizar:", numeric_columns)

        col_graph1, col_graph2 = st.columns(2)

        with col_graph1:
            st.subheader("Histograma + KDE")
            fig_hist, ax_hist = plt.subplots()
            sns.histplot(df[variable], kde=True, ax=ax_hist, color="skyblue")
            ax_hist.set_title(f"Distribución de {variable}")
            st.pyplot(fig_hist)

        with col_graph2:
            st.subheader("Boxplot (Outliers)")
            fig_box, ax_box = plt.subplots()
            sns.boxplot(y=df[variable], ax=ax_box, color="lightcoral")
            ax_box.set_title(f"Detección de Outliers en {variable}")
            st.pyplot(fig_box)

        # Preguntas obligatorias de la app
        st.markdown("### 📝 Análisis de la Distribución")
        c1, c2 = st.columns(2)
        with c1:
            st.radio("¿La distribución parece normal?", ["Sí", "No", "Incierto"], key="norm_check")
        with c2:
            st.text_input("¿Hay sesgo o outliers visibles?", placeholder="Ej: Sesgo a la derecha con 2 outliers...", key="bias_check")
            
    else:
        st.warning("El archivo subido no contiene columnas numéricas.")

    # --- ESPACIO PARA MÓDULO 3: PRUEBA Z (PRÓXIMO PASO) ---
    st.divider()
    st.header("3. Prueba de Hipótesis (Z-Test)")
    st.info("Aquí implementaremos la lógica matemática de la Prueba Z en el siguiente commit.")

else:
    # Estado inicial de la app
    st.info("Esperando archivo CSV para iniciar el análisis...")