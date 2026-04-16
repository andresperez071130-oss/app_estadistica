import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
import os
from dotenv import load_dotenv

# 1. CARGA DE CONFIGURACIÓN Y LLAVES (Usando tus recursos actuales)
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

# 2. CONFIGURACIÓN DE INTERFAZ (Estilo Oscuro Pro)
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

# 3. FUNCIONES LÓGICAS
def call_gemini(api_key, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, json=payload, timeout=30)
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except: return "❌ Error al conectar con la IA. Revisa tu archivo .env"

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

# 4. BARRA LATERAL (SIDEBAR)
with st.sidebar:
    st.title("⚙️ Configuración")
    source = st.radio("Fuente de Datos:", ["Sintéticos", "Subir CSV"])
    
    df = None
    if source == "Subir CSV":
        uploaded = st.file_uploader("Carga tu archivo CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
    else:
        st.subheader("Generador de Datos")
        n_val = st.number_input("Tamaño de muestra (n):", min_value=30, value=100)
        mu_gen = st.number_input("Media a generar:", value=50.0)
        sigma_gen = st.number_input("Desviación a generar:", value=10.0)
        if st.button("🎲 Generar"):
            df = pd.DataFrame({"Valor": np.random.normal(mu_gen, sigma_gen, n_val)})

# 5. CONTENIDO PRINCIPAL
if df is not None:
    # Selección de variable
    cols = df.select_dtypes(include=np.number).columns.tolist()
    var = st.selectbox("Variable para el estudio:", cols)
    data = df[var].dropna()
    n = len(data)

    tab1, tab2, tab3 = st.tabs(["📈 Visualización", "🧪 Prueba Z", "🤖 Asistente IA"])

    with tab1:
        st.header("Análisis de Distribución")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Histograma**")
            fig, ax = plt.subplots(); sns.histplot(data, color="#63b3ed", ax=ax); st.pyplot(fig)
        with col2:
            st.write("**Densidad (KDE)**")
            fig, ax = plt.subplots(); sns.kdeplot(data, fill=True, color="#9f7aea", ax=ax); st.pyplot(fig)
        with col3:
            st.write("**Caja (Boxplot)**")
            fig, ax = plt.subplots(); sns.boxplot(y=data, color="#63b3ed", ax=ax); st.pyplot(fig)

    with tab2:
        st.header("Inferencia: Prueba Z")
        # Validación de n >= 30 (Requisito Rúbrica)
        if n < 30:
            st.error(f"⚠️ Tamaño de muestra insuficiente (n={n}). Se requiere n ≥ 30.")
        
        c1, c2 = st.columns(2)
        with c1:
            mu0 = st.number_input("Hipótesis Nula (μ₀):", value=float(data.mean()))
            tail = st.selectbox("Hipótesis Alternativa (H₁):", ["Bilateral", "Superior", "Inferior"])
        with c2:
            sigma_pob = st.number_input("Desv. Estándar Poblacional (σ):", value=float(data.std()))
            alpha = st.select_slider("Significancia (α):", options=[0.01, 0.05, 0.10], value=0.05)

        z, p, rej, crit = calculate_z_test(data.mean(), mu0, sigma_pob, n, alpha, tail)
        
        st.markdown(f"""
        <div class='stat-card'>
            <h3>Decisión Estadística</h3>
            <p style='color: {"#fc4242" if rej else "#48bb78"};'>
                {'RECHAZAR H₀' if rej else 'NO RECHAZAR H₀'}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Gráfica de Curva Normal
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
        
        ax_z.axvline(z, color='#9f7aea', ls='--', lw=2, label=f'Z={z:.2f}')
        ax_z.legend(); st.pyplot(fig_z)
    with tab3:
        st.header("Consultoría Experta")
        if st.button("🚀 Analizar con Gemini"):
            if gemini_key:
                # Datos para la IA
                resumen = f"Z={z:.4f}, p={p:.4f}, n={n}, alpha={alpha}, H1={tail}, decision={'Rechazar' if rej else 'No rechazar'}"
                
                # Prompt para explicación sencilla y extensa
                prompt = f"""
                Actúa como un profesor de estadística didáctico. Explica estos resultados a alguien 
                que no sabe estadística usando un lenguaje sencillo y ameno.
                
                RESULTADOS: {resumen}.
                
                ESTRUCTURA:
                1. Conclusión General: Explica si hubo un cambio real o no.
                2. Importancia de los datos: Menciona por qué confiar en n={n} y la desviación.
                Redacta al menos dos párrafos explicativos.
                """
                
                with st.spinner("Preparando explicación sencilla..."):
                    explicacion = call_gemini(gemini_key, prompt)
                    st.markdown("### 💡 Interpretación para Todos")
                    st.write(explicacion)
            else:
                st.error("No se detectó la llave en el archivo .env")
    
else:
    st.info("👈 Configura la fuente de datos en la barra lateral para comenzar.")