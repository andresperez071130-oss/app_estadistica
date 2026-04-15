import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuración de la página
st.set_page_config(page_title="App de Estadística Interactiva", layout="wide")

st.title("📊 Aplicación de Análisis Estadístico")
st.markdown("Herramienta interactiva para visualización de distribuciones y pruebas de hipótesis.")

# --- MÓDULO 1: ADQUISICIÓN DE DATOS ---
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
        n_puntos = st.number_input("Tamaño de muestra (n):", min_value=30, value=100)
    with col_s2:
        media_s = st.number_input("Media deseada:", value=50.0)
    with col_s3:
        desv_s = st.number_input("Desviación estándar:", value=10.0, min_value=0.1)
    
    if st.button("Generar Datos"):
        datos_sinteticos = np.random.normal(media_s, desv_s, n_puntos)
        df = pd.DataFrame({'Variable_Sintetica': datos_sinteticos})

if df is not None:
    # --- MÓDULO 2: VISUALIZACIÓN ---
    st.divider()
    st.header("2. Visualización de Distribuciones")
    
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    variable = st.selectbox("Selecciona la variable a analizar:", numeric_columns)
    data = df[variable].dropna()

    # Gráficas en 3 columnas independientes
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

    # Diagnóstico automático de la variable elegida
    st.subheader(f"🔍 Diagnóstico Automático: {variable}")
    m1, m2, m3 = st.columns(3)
    
    # 1. Normalidad
    _, p_norm = stats.shapiro(data) if len(data) > 3 else (0, 0)
    m1.metric("¿Es Normal?", "Sí ✅" if p_norm > 0.05 else "No ❌")
    
    # 2. Sesgo
    skew = data.skew()
    m2.metric("Sesgo", "Positivo" if skew > 0.5 else "Negativo" if skew < -0.5 else "Simétrico")
    
    # 3. Outliers
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR = Q3 - Q1
    n_outliers = len(data[(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))])
    m3.metric("Outliers", n_outliers)

    # --- MÓDULO 3: PRUEBA Z ---
    st.divider()
    st.header("3. Prueba de Hipótesis (Z-Test)")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        mu_0 = st.number_input("Hipótesis Nula (H0: μ = ?):", value=float(data.mean()))
        tipo_prueba = st.selectbox("Hipótesis Alternativa (H1):", ["Bilateral (≠)", "Cola Derecha (>)", "Cola Izquierda (<)"])
    with col_p2:
        alpha = st.slider("Nivel de significancia (α):", 0.01, 0.10, 0.05)
        sigma = st.number_input("Desviación estándar (σ) conocida:", value=float(data.std()))

    # Cálculos
    x_bar = data.mean()
    n = len(data)
    z_stat = (x_bar - mu_0) / (sigma / np.sqrt(n))

    if tipo_prueba == "Bilateral (≠)":
        z_crit = stats.norm.ppf(1 - alpha/2)
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        rechazo = abs(z_stat) > z_crit
    elif tipo_prueba == "Cola Derecha (>)":
        z_crit = stats.norm.ppf(1 - alpha)
        p_val = 1 - stats.norm.cdf(z_stat)
        rechazo = z_stat > z_crit
    else: # Cola Izquierda
        z_crit = stats.norm.ppf(alpha)
        p_val = stats.norm.cdf(z_stat)
        rechazo = z_stat < z_crit

    # Mostrar Resultados Numéricos
    st.subheader("Resultados")
    res1, res2, res3 = st.columns(3)
    res1.metric("Z calculado", f"{z_stat:.4f}")
    res2.metric("P-Value", f"{p_val:.4f}")
    res3.metric("Decisión", "RECHAZAR H0" if rechazo else "NO RECHAZAR H0")

    # Gráfica de la Región Crítica
    fig_z, ax_z = plt.subplots(figsize=(10, 4))
    x_axis = np.linspace(-4, 4, 1000)
    y_axis = stats.norm.pdf(x_axis, 0, 1)
    ax_z.plot(x_axis, y_axis, color='black')

    if tipo_prueba == "Bilateral (≠)":
        ax_z.fill_between(x_axis, y_axis, where=(abs(x_axis) > z_crit), color='red', alpha=0.5, label='Región de Rechazo')
    elif tipo_prueba == "Cola Derecha (>)":
        ax_z.fill_between(x_axis, y_axis, where=(x_axis > z_crit), color='red', alpha=0.5, label='Región de Rechazo')
    else:
        ax_z.fill_between(x_axis, y_axis, where=(x_axis < z_crit), color='red', alpha=0.5, label='Región de Rechazo')

    ax_z.axvline(z_stat, color='blue', linestyle='--', label=f'Z de tu muestra ({z_stat:.2f})')
    ax_z.set_title("Distribución Normal Estándar y Región Crítica")
    ax_z.legend()
    st.pyplot(fig_z)

else:
    st.info("Sube un CSV o genera datos sintéticos para habilitar los módulos.")