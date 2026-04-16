import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
import os
from dotenv import load_dotenv

# 1. INICIALIZACIÓN Y SEGURIDAD
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="StatLab Pro - UPChiapas", layout="wide")

# Estilos CSS de alto nivel (Dark Mode)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700&family=JetBrains+Mono&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0f0f1a; color: #e8e8f0; }
.stat-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(99,179,237,0.3); border-radius: 12px; padding: 20px; }
h1 { color: #63b3ed !important; }
</style>
""", unsafe_allow_html=True)

# 2. FUNCIONES DE LÓGICA
def call_gemini(api_key, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, json=payload, timeout=30)
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except: return "❌ Error al conectar con la IA. Verifica tu llave en el .env"

def calculate_z_test(x_bar, mu0, sigma, n, alpha, tail):
    # Error corregido: Cálculo preciso del error estándar y Z
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
    else: # Inferior
        p = stats.norm.cdf(z)
        crit = stats.norm.ppf(alpha)
        rej = z < crit
    return z, p, rej, crit

# 3. BARRA LATERAL COMPLETA
with st.sidebar:
    st.title("⚙️ Configuración")
    source = st.radio("Fuente de Datos:", ["Sintéticos", "Subir CSV"])
    
    if source == "Subir CSV":
        uploaded = st.file_uploader("Archivo CSV", type=["csv"])
        if uploaded:
            st.session_state['df'] = pd.read_csv(uploaded)
    else:
        st.subheader("Generador de Datos")
        dist_type = st.selectbox("Distribución:", ["Normal", "Sesgada", "Uniforme", "Bimodal"])
        n_val = st.number_input("Tamaño (n):", min_value=30, value=100)
        mu_s = st.number_input("Media (μ):", value=50.0)
        sigma_s = st.number_input("Desviación (σ):", value=10.0)
        
        if st.button("🎲 Generar Dataset"):
            if dist_type == "Normal":
                data = np.random.normal(mu_s, sigma_s, n_val)
            elif dist_type == "Sesgada":
                data = np.random.lognormal(mean=2, sigma=0.5, size=n_val)
            elif dist_type == "Uniforme":
                data = np.random.uniform(mu_s-10, mu_s+10, n_val)
            else: # Bimodal
                d1 = np.random.normal(mu_s-15, 5, n_val//2)
                d2 = np.random.normal(mu_s+15, 5, n_val//2)
                data = np.concatenate([d1, d2])
            
            st.session_state['df'] = pd.DataFrame({"Valor_Generado": data})
            st.success(f"Dataset {dist_type} generado.")

# 4. CUERPO PRINCIPAL
if 'df' in st.session_state:
    df = st.session_state['df']
    var = st.selectbox("Variable a analizar:", df.select_dtypes(include=np.number).columns)
    data = df[var].dropna()
    n = len(data)

    tab1, tab2, tab3 = st.tabs(["📊 Visualización", "🧪 Prueba Z", "🤖 Asistente IA"])

    with tab1:
        st.header("Distribución Detallada")
        # Visualización en 3 segmentos como pediste
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Histograma**")
            fig1, ax1 = plt.subplots(); sns.histplot(data, color="#63b3ed", ax=ax1); st.pyplot(fig1)
        with c2:
            st.write("**Curva de Densidad**")
            fig2, ax2 = plt.subplots(); sns.kdeplot(data, fill=True, color="#9f7aea", ax=ax2); st.pyplot(fig2)
        with c3:
            st.write("**Análisis de Outliers**")
            fig3, ax3 = plt.subplots(); sns.boxplot(y=data, color="#63b3ed", ax=ax3); st.pyplot(fig3)

    with tab2:
        st.header("Inferencia con Varianza Conocida")
        if n < 30: st.warning("⚠️ Nota: n < 30. Se recomienda precaución con la Prueba Z.")
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            mu0 = st.number_input("H₀ (μ =)", value=float(data.mean()))
            tail = st.selectbox("H₁ (Tipo):", ["Bilateral", "Superior", "Inferior"])
        with col_p2:
            sigma_p = st.number_input("σ Poblacional:", value=float(data.std()))
            alpha = st.select_slider("Significancia (α):", options=[0.01, 0.05, 0.10], value=0.05)

        z, p, rej, crit = calculate_z_test(data.mean(), mu0, sigma_p, n, alpha, tail)
        
        st.markdown(f"""<div class='stat-card'><h3>Decisión: {'RECHAZAR' if rej else 'NO RECHAZAR'} H₀</h3>
                    <p>Z = {z:.4f} | p-value = {p:.4f} | n = {n}</p></div>""", unsafe_allow_html=True)
        
        # Gráfica de la segunda sección (mantiene el estilo anterior)
        x = np.linspace(-4, 4, 1000)
        y = stats.norm.pdf(x)
        fig_z, ax_z = plt.subplots(figsize=(10, 4))
        ax_z.plot(x, y, color='#63b3ed', lw=2)
        if tail == "Bilateral":
            ax_z.fill_between(x, y, where=(abs(x) > crit), color='#fc4242', alpha=0.4)
        elif tail == "Superior":
            ax_z.fill_between(x, y, where=(x > crit), color='#fc4242', alpha=0.4)
        else:
            ax_z.fill_between(x, y, where=(x < crit), color='#fc4242', alpha=0.4)
        ax_z.axvline(z, color='#9f7aea', ls='--', label=f'Z={z:.2f}'); ax_z.legend()
        st.pyplot(fig_z)

    with tab3:
        st.header("Consultoría Experta")
        if st.button("🚀 Analizar con Gemini"):
            if gemini_key:
                resumen = f"Z={z:.4f}, p={p:.4f}, n={n}, alpha={alpha}, H1={tail}, decision={'Rechazar' if rej else 'No rechazar'}"
                prompt = f"""Actúa como un profesor didáctico. Explica estos resultados a alguien que no sabe estadística. 
                Sé breve y con emoji. Resultados: {resumen}."""
                with st.spinner("Redactando conclusión..."):
                    st.info(call_gemini(gemini_key, prompt))
            else: st.error("Falta API Key en el .env")
else:
    st.info("👈 Genera datos o sube un CSV en la barra lateral.")