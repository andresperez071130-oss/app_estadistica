import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuración de la página
st.set_page_config(page_title="App de Estadística - UPChiapas", layout="wide")

st.title("📊 Asistente de Análisis Estadístico Automático")
st.markdown("Esta versión detecta automáticamente normalidad, sesgo y outliers.")

# --- MÓDULO 1: CARGA DE DATOS ---
st.header("1. Carga de Datos")
uploaded_file = st.file_uploader("Sube tu archivo CSV aquí", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Vista previa y Resumen")
    col_pre1, col_pre2 = st.columns(2)
    with col_pre1:
        st.write("Primeros registros:")
        st.dataframe(df.head())
    with col_pre2:
        st.write("Estadísticas descriptivas:")
        st.write(df.describe())
    
    # --- MÓDULO 2: VISUALIZACIÓN Y ANÁLISIS AUTOMÁTICO ---
    st.divider()
    st.header("2. Visualización y Detección Automática")

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if numeric_columns:
        variable = st.selectbox("Selecciona una variable para analizar:", numeric_columns)
        data = df[variable].dropna() # Limpiamos nulos para los cálculos

        col_graph1, col_graph2 = st.columns(2)

        with col_graph1:
            st.subheader("Histograma + KDE")
            fig_hist, ax_hist = plt.subplots()
            sns.histplot(data, kde=True, ax=ax_hist, color="skyblue")
            st.pyplot(fig_hist)

        with col_graph2:
            st.subheader("Boxplot (Outliers)")
            fig_box, ax_box = plt.subplots()
            sns.boxplot(y=data, ax=ax_box, color="lightcoral")
            st.pyplot(fig_box)

        # --- LÓGICA DE DETECCIÓN AUTOMÁTICA ---
        st.subheader(f"🔍 Diagnóstico Automático de: {variable}")
        
        # 1. Prueba de Normalidad (Shapiro-Wilk)
        # Si p-value > 0.05, se asume normalidad
        stat, p_norm = stats.shapiro(data) if len(data) > 3 else (0, 0)
        es_normal = "Sí (Distribución Normal)" if p_norm > 0.05 else "No (No Normal)"

        # 2. Detección de Sesgo (Skewness)
        skewness = data.skew()
        if skewness > 0.5:
            tipo_sesgo = "Positivo (Derecha)"
        elif skewness < -0.5:
            tipo_sesgo = "Negativo (Izquierda)"
        else:
            tipo_sesgo = "Simétrico"

        # 3. Detección de Outliers (Método IQR)
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))]
        conteo_outliers = len(outliers)

        # Mostrar resultados en métricas
        m1, m2, m3 = st.columns(3)
        m1.metric("¿Es Normal?", es_normal)
        m2.metric("Sesgo detectado", tipo_sesgo)
        m3.metric("Outliers encontrados", conteo_outliers)

        # Comparativa sugerida por la actividad
        st.info(f"**Análisis:** La variable '{variable}' tiene un coeficiente de asimetría de {skewness:.2f}. "
                f"Se detectaron {conteo_outliers} valores atípicos.")

    else:
        st.warning("El archivo no contiene columnas numéricas.")

    # --- PRÓXIMO PASO: PRUEBA Z ---
    st.divider()
    st.header("3. Prueba de Hipótesis (Z-Test)")
    st.write("Pendiente de implementación: Definición de H0, H1 y cálculos.")

else:
    st.info("Esperando archivo CSV...")