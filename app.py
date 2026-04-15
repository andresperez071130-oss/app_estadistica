import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuración de la página
st.set_page_config(page_title="App de Estadística - UPChiapas", layout="wide")

st.title("📊 Asistente de Análisis Estadístico Automatizado")

# --- MÓDULO 1: ADQUISICIÓN DE DATOS (CSV o SINTÉTICOS) ---
st.header("1. Carga o Generación de Datos")

opcion_datos = st.radio("Selecciona la fuente de datos:", ["Subir CSV", "Generar Datos Sintéticos"])

df = None

if opcion_datos == "Subir CSV":
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    st.subheader("Configuración de Datos Sintéticos")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        n_puntos = st.number_input("Tamaño de muestra (n):", min_value=10, max_value=10000, value=100)
    with col_s2:
        media_s = st.number_input("Media deseada:", value=50.0)
    with col_s3:
        desv_s = st.number_input("Desviación estándar:", value=10.0, min_value=0.1)
    
    if st.button("Generar Datos"):
        datos_sinteticos = np.random.normal(media_s, desv_s, n_puntos)
        df = pd.DataFrame({'Variable_Sintetica': datos_sinteticos})
        st.success("Datos generados correctamente")

# --- PROCESAMIENTO SI HAY DATOS ---
if df is not None:
    st.divider()
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    # --- MÓDULO 2: VISUALIZACIÓN DE DISTRIBUCIONES (Separadas) ---
    st.header("2. Visualización de Distribuciones")
    
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    variable = st.selectbox("Selecciona la variable a analizar:", numeric_columns)
    data = df[variable].dropna()

    # Gráficas en 3 columnas
    g1, g2, g3 = st.columns(3)

    with g1:
        st.subheader("Histograma")
        fig1, ax1 = plt.subplots()
        sns.histplot(data, kde=False, ax=ax1, color="skyblue")
        st.pyplot(fig1)

    with g2:
        st.subheader("Gráfica KDE")
        fig2, ax2 = plt.subplots()
        sns.kdeplot(data, ax=ax2, color="green", fill=True)
        st.pyplot(fig2)

    with g3:
        st.subheader("Boxplot")
        fig3, ax3 = plt.subplots()
        sns.boxplot(y=data, ax=ax3, color="lightcoral")
        st.pyplot(fig3)

    # --- DIAGNÓSTICO AUTOMÁTICO ---
    st.subheader(f"🔍 Diagnóstico Automático de: {variable}")
    
    # 1. Normalidad (Shapiro-Wilk)
    stat, p_norm = stats.shapiro(data) if len(data) > 3 else (0, 0)
    es_normal = "Sí ✅" if p_norm > 0.05 else "No ❌"

    # 2. Sesgo (Skewness)
    skewness = data.skew()
    if skewness > 0.5: tipo_sesgo = "Positivo (Derecha)"
    elif skewness < -0.5: tipo_sesgo = "Negativo (Izquierda)"
    else: tipo_sesgo = "Simétrico"

    # 3. Outliers (IQR)
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR = Q3 - Q1
    n_outliers = len(data[(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))])

    m1, m2, m3 = st.columns(3)
    m1.metric("¿Es Normal?", es_normal, f"p={p_norm:.4f}")
    m2.metric("Sesgo", tipo_sesgo, f"skew={skewness:.2f}")
    m3.metric("Outliers", n_outliers)

    # --- PRÓXIMO PASO: PRUEBA Z ---
    st.divider()
    st.header("3. Prueba de Hipótesis (Z-Test)")
    st.info("Aquí configuraremos H0, H1 y el nivel de significancia en el siguiente paso.")

else:
    st.info("Sube un CSV o genera datos sintéticos para continuar.")