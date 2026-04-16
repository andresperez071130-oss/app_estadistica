import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests

# 1. Configuración de Estilo y UI de Alto Nivel
st.set_page_config(page_title="StatLab Pro", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700&family=JetBrains+Mono&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
code { font-family: 'JetBrains Mono', monospace; }
.stApp { background: #0f0f1a; color: #e8e8f0; }
.stat-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
}
h1 { color: #63b3ed !important; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# 2. Funciones Modulares de Lógica
def call_gemini(api_key, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, json=payload, timeout=30)
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except: return "❌ Error al conectar con la IA."

def calculate_z_test(x_bar, mu0, sigma, n, alpha, tail):
    se = sigma / np.sqrt(n)
    z = (x_bar - mu0) / se
    if tail == "Bilateral":
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        crit = stats.norm.ppf(1 - alpha / 2)
        reject = abs(z) > crit
    elif tail == "Superior":
        p = 1 - stats.norm.cdf(z)
        crit = stats.norm.ppf(1 - alpha)
        reject = z > crit
    else:
        p = stats.norm.cdf(z)
        crit = stats.norm.ppf(alpha)
        reject = z < crit
    return z, p, reject, crit

# 3. Sidebar y Carga de Datos
with st.sidebar:
    st.title("⚙️ Configuración")
    # Seguridad: Lee desde st.secrets
    gemini_key = st.secrets["GEMINI_API_KEY"] 
    source = st.radio("Datos:", ["Sintéticos", "Subir CSV"])
    df = None
    if source == "Subir CSV":
        uploaded = st.file_uploader("CSV", type=["csv"])
        if uploaded: df = pd.read_csv(uploaded)
    else:
        n_val = st.number_input("n (Muestra):", min_value=30, value=100)
        df = pd.DataFrame({"Valor": np.random.normal(50, 10, n_val)})

if df is not None:
    var = st.selectbox("Selecciona Variable:", df.select_dtypes(include=np.number).columns)
    data = df[var].dropna()
    n = len(data)

    tab1, tab2, tab3 = st.tabs(["📈 Visualización", "🧪 Prueba Z", "🤖 Asistente IA"])

    # --- TAB 1: Visualización ---
    with tab1:
        st.header("Análisis de Distribución")
        col1, col2, col3 = st.columns(3)
        with col1:
            fig, ax = plt.subplots(); sns.histplot(data, color="#63b3ed", ax=ax); st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(); sns.kdeplot(data, fill=True, color="#9f7aea", ax=ax); st.pyplot(fig)
        with col3:
            fig, ax = plt.subplots(); sns.boxplot(y=data, color="#63b3ed", ax=ax); st.pyplot(fig)

    # --- TAB 2: Prueba Z ---
    with tab2:
        st.header("Inferencia Estadística")
        if n < 30: st.error("⚠️ n debe ser ≥ 30 para una Prueba Z válida.")
        
        c1, c2 = st.columns(2)
        with c1:
            mu0 = st.number_input("H₀ (Media Hipotética):", value=float(data.mean()))
            tail = st.selectbox("H₁ (Cola):", ["Bilateral", "Superior", "Inferior"])
        with c2:
            sigma = st.number_input("σ (Desv. Est. Poblacional):", value=float(data.std()))
            alpha = st.slider("Significancia (α):", 0.01, 0.10, 0.05)

        z, p, rej, crit = calculate_z_test(data.mean(), mu0, sigma, n, alpha, tail)
        
        st.markdown(f"""<div class='stat-card'><h3>Resultado</h3><p>{'RECHAZAR H₀' if rej else 'NO RECHAZAR H₀'}</p></div>""", unsafe_allow_html=True)
        
        # Gráfica de curva normal y sombreado
        x = np.linspace(-4, 4, 1000)
        y = stats.norm.pdf(x)
        fig_z, ax_z = plt.subplots(figsize=(10, 4))
        ax_z.plot(x, y, color='#63b3ed')
        if tail == "Bilateral":
            ax_z.fill_between(x, y, where=(abs(x) > crit), color='#fc4242', alpha=0.5)
        elif tail == "Superior":
            ax_z.fill_between(x, y, where=(x > crit), color='#fc4242', alpha=0.5)
        else:
            ax_z.fill_between(x, y, where=(x < crit), color='#fc4242', alpha=0.5)
        ax_z.axvline(z, color='#9f7aea', linestyle='--', label=f'Z={z:.2f}')
        st.pyplot(fig_z)

    # --- TAB 3: Asistente IA ---
    with tab3:
        st.header("Interpretación con Gemini")
        if st.button("🚀 Consultar IA"):
            # Resumen técnico (sin datos crudos)
            resumen = f"Z={z:.4f}, p={p:.4f}, n={n}, alpha={alpha}, decision={'Rechazar' if rej else 'No rechazar'}"
            prompt = f"Actúa como estadístico. Explica estos resultados de una prueba Z: {resumen}. Evalúa supuestos."
            st.write(call_gemini(gemini_key, prompt))