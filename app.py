import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
import os
from dotenv import load_dotenv

# 1. CONFIGURACIÓN INICIAL
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="StatLab Pro Multi-Var", layout="wide")

# Estilos CSS (Mantenemos tu estética Dark Pro)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700&family=JetBrains+Mono&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0f0f1a; color: #e8e8f0; }
.stat-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(99,179,237,0.3); border-radius: 12px; padding: 20px; }
</style>
""", unsafe_allow_html=True)

# 2. FUNCIONES DE APOYO
def call_gemini(api_key, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, json=payload, timeout=30)
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except: return "❌ Error de conexión. Revisa tu .env o la cuota de la API."

def calculate_z_test(x_bar, mu0, sigma, n, alpha, tail):
    se = sigma / np.sqrt(n)
    z = (x_bar - mu0) / se
    if tail == "Bilateral":
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        crit = stats.norm.ppf(1 - alpha / 2)
        rej = abs(z) > crit
    elif tail == "Superior":
        p = 1 - stats.norm.cdf(z)
        crit = stats.norm.ppf(1 - alpha)
        rej = z > crit
    else:
        p = stats.norm.cdf(z)
        crit = stats.norm.ppf(alpha)
        rej = z < crit
    return z, p, rej, crit

# 3. BARRA LATERAL: GENERACIÓN MULTI-VARIABLE
with st.sidebar:
    st.title("⚙️ Configuración")
    source = st.radio("Fuente de Datos:", ["Sintéticos", "Subir CSV"])
    
    if source == "Subir CSV":
        uploaded = st.file_uploader("Archivo CSV", type=["csv"])
        if uploaded:
            st.session_state['df'] = pd.read_csv(uploaded)
    else:
        st.subheader("Generador Multi-Variable")
        n_val = st.number_input("Tamaño (n):", min_value=30, value=100)
        if st.button("🎲 Generar Dataset Completo"):
            # Creamos un dataset con 3 variables diferentes para analizar
            data_dict = {
                "Productividad": np.random.normal(80, 5, n_val),
                "Eficiencia_Tiempo": np.random.normal(50, 10, n_val),
                "Costo_Operativo": np.random.normal(120, 15, n_val)
            }
            st.session_state['df'] = pd.DataFrame(data_dict)
            st.success("¡Dataset generado con 3 variables!")

# 4. LÓGICA PRINCIPAL (Validación de persistencia)
if 'df' in st.session_state:
    df = st.session_state['df']
    cols = df.select_dtypes(include=np.number).columns.tolist()
    
    st.subheader("Selección de Análisis")
    var_sel = st.selectbox("¿Qué variable deseas estudiar hoy?", cols)
    data = df[var_sel].dropna()
    n = len(data)

    tab1, tab2, tab3 = st.tabs(["📈 Visualización", "🧪 Prueba Z", "🤖 Asistente IA"])

    with tab1:
        st.header(f"Distribución: {var_sel}")
        c1, c2 = st.columns(2)
        with c1:
            fig1, ax1 = plt.subplots(); sns.histplot(data, kde=True, color="#63b3ed", ax=ax1); st.pyplot(fig1)
        with c2:
            fig2, ax2 = plt.subplots(); sns.boxplot(x=data, color="#9f7aea", ax=ax2); st.pyplot(fig2)

    with tab2:
        st.header("Cálculos de Inferencia")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            mu0 = st.number_input("μ₀ (Hipótesis Nula):", value=float(data.mean()))
            tail = st.selectbox("Tipo de Prueba:", ["Bilateral", "Superior", "Inferior"])
        with col_p2:
            sigma_p = st.number_input("σ (Poblacional conocida):", value=float(data.std()))
            alpha = st.select_slider("Alfa (α):", options=[0.01, 0.05, 0.10], value=0.05)

        z, p, rej, crit = calculate_z_test(data.mean(), mu0, sigma_p, n, alpha, tail)
        
        st.markdown(f"""<div class='stat-card'><h3>Decisión: {'RECHAZAR' if rej else 'NO RECHAZAR'} H₀</h3>
                    <p>Z = {z:.4f} | p-value = {p:.4f}</p></div>""", unsafe_allow_html=True)

    with tab3:
        st.header("Reporte Experto")
        if st.button("🚀 Generar Reporte con IA"):
            if gemini_key:
                resumen = f"Var={var_sel}, Z={z:.4f}, p={p:.4f}, n={n}, decision={'Rechazar' if rej else 'No rechazar'}"
                prompt = f"Explica estos resultados de forma sencilla y breve a un no experto: {resumen}."
                
                with st.spinner("La IA está redactando..."):
                    respuesta = call_gemini(gemini_key, prompt)
                    st.info(respuesta)
            else:
                st.error("Falta API Key en el archivo .env")
else:
    st.info("👈 Selecciona una fuente de datos para activar el panel de análisis.")