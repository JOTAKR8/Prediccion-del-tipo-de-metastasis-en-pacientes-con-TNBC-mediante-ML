# -*- coding: utf-8 -*-

import pandas as pd
import streamlit as st
import pickle


# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="MediPredict | Cáncer de Mama",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# ESTILOS — CORPORATIVO SALUD
# =========================================================

st.markdown("""
<style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
    }

    :root {
        --navy: #0b2545;
        --navy-dark: #081a33;
        --teal: #0f9d8b;
        --teal-dark: #0b7c6e;
        --bg: #f7f9fb;
        --line: #e1e6ec;
        --ink: #12213b;
        --sub: #63728a;
    }

    .stApp {
        background: var(--bg);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 0rem;
        padding-bottom: 3rem;
    }

    #MainMenu, footer, header {visibility: hidden;}

    /* ---------- Barra superior ---------- */
    .topbar {
        background: var(--navy);
        color: #ffffff;
        padding: 16px 40px;
        margin: 0 -100px 0 -100px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .topbar .brand {
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 0.04em;
    }

    .topbar .tag {
        font-size: 11.5px;
        background: rgba(255,255,255,0.12);
        padding: 4px 12px;
        border-radius: 4px;
        color: #cdd6e3;
    }

    /* ---------- Hero ---------- */
    .hero {
        background: linear-gradient(180deg, var(--navy) 0%, var(--navy-dark) 100%);
        color: #ffffff;
        padding: 34px 40px 46px;
        margin: 0 -100px 0 -100px;
    }

    .hero h1 {
        font-size: 26px;
        font-weight: 700;
        margin: 0 0 8px 0;
    }

    .hero p {
        font-size: 13.5px;
        color: #c9d3e0;
        max-width: 560px;
        margin: 0;
        line-height: 1.6;
    }

    /* ---------- Contenedor que "sube" sobre el hero ---------- */
    .lift {
        margin-top: -22px;
    }

    /* ---------- Tarjetas ---------- */
    .card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 24px 26px;
        box-shadow: 0 2px 10px rgba(11, 37, 69, 0.07);
        border-top: 3px solid var(--teal);
        margin-bottom: 20px;
    }

    .card h3 {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--navy);
        margin: 0 0 16px 0;
        font-weight: 700;
    }

    .card p.note {
        font-size: 12.5px;
        color: var(--sub);
        margin-top: 4px;
    }

    /* ---------- Banner informativo ---------- */
    .info-banner {
        background: #ffffff;
        border: 1px solid var(--line);
        border-left: 4px solid var(--teal);
        border-radius: 8px;
        padding: 14px 20px;
        font-size: 13.5px;
        color: var(--ink);
        margin-bottom: 22px;
    }

    /* ---------- Resultado ---------- */
    .resultado-box {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 26px;
        text-align: center;
        margin: 18px 0 26px 0;
        box-shadow: 0 2px 10px rgba(11, 37, 69, 0.07);
    }

    .resultado-box .label {
        font-size: 11.5px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--sub);
        margin-bottom: 8px;
        font-weight: 700;
    }

    .resultado-box .valor {
        font-size: 22px;
        font-weight: 800;
        color: var(--navy);
    }

    /* ---------- Métricas ---------- */
    div[data-testid="stMetric"] {
        background-color: #fbfcfd;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 12px 14px;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 12px !important;
        color: var(--sub) !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    div[data-testid="stMetricValue"] {
        font-size: 18px !important;
        color: var(--navy) !important;
        font-weight: 700 !important;
    }

    /* ---------- Inputs ---------- */
    div[data-baseweb="select"] > div {
        border-radius: 6px !important;
        border-color: var(--line) !important;
    }

    label, .stSelectbox label {
        font-size: 12.5px !important;
        color: var(--sub) !important;
        font-weight: 600 !important;
    }

    /* ---------- Botón ---------- */
    .stButton > button {
        background: var(--teal);
        color: white;
        border-radius: 6px;
        height: 50px;
        font-size: 15px;
        font-weight: 700;
        border: none;
    }

    .stButton > button:hover {
        background: var(--teal-dark);
        color: white;
    }

    hr {
        margin: 1.6rem 0 !important;
        border-color: var(--line) !important;
    }

    .footer-note {
        text-align: center;
        color: var(--sub);
        font-size: 12.5px;
        margin-top: 6px;
    }

    /* ---------- Subgrupos dentro de una tarjeta ---------- */
    .group-title {
        font-size: 11.5px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--teal-dark);
        font-weight: 700;
        margin: 18px 0 10px 0;
    }

    .group-title.first {
        margin-top: 2px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# CARGAMOS EL MODELO
# =========================================================

modelo = pickle.load(open("modelo_random_forest.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))


# =========================================================
# CARGAMOS DATOS ZIP
# =========================================================

zip_data = pd.read_csv("zip_completo.csv")

columnas_numericas = [c for c in zip_data.columns if c != "patient_zip3"]

for columna in columnas_numericas:
    zip_data[columna] = pd.to_numeric(zip_data[columna], errors="coerce")

zip_data["patient_zip3"] = pd.to_numeric(zip_data["patient_zip3"], errors="coerce")

zips_disponibles = sorted(
    zip_data["patient_zip3"].dropna().astype(int).unique()
)


# =========================================================
# BARRA SUPERIOR + HERO
# =========================================================

st.markdown("""
<div class="topbar">
    <div class="brand">MEDIPREDICT</div>
    <div class="tag">Sistema académico</div>
</div>
<div class="hero">
    <h1>Predicción del tipo de metástasis en cáncer de mama</h1>
    <p>Herramienta de apoyo a la decisión clínica que combina información
    clínica del paciente con variables sociodemográficas de su zona
    geográfica mediante un modelo de Machine Learning.</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# CONTENIDO PRINCIPAL
# =========================================================

st.markdown('<div class="lift">', unsafe_allow_html=True)

st.markdown("""
<div class="info-banner">
    Ingrese los datos básicos del paciente. El sistema utilizará el
    código ZIP3 para obtener automáticamente las variables
    sociodemográficas asociadas a esa zona.
</div>
""", unsafe_allow_html=True)


col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <h3>Información del paciente</h3>
    """, unsafe_allow_html=True)

    zip3 = st.selectbox("Código ZIP3", zips_disponibles)
    Edad = st.selectbox("Edad del paciente", list(range(0, 92)), index=50)
    Payer = st.selectbox(
        "Tipo de pagador",
        ["MEDICAID", "COMMERCIAL", "MEDICARE ADVANTAGE"]
    )

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>Información clínica</h3>
    """, unsafe_allow_html=True)

    Diagnostico = st.selectbox(
        "Código de diagnóstico de cáncer de mama",
        [
            "C50919", "C50411", "C50112", "C50212", "1749", "C50912",
            "C50512", "1744", "C50412", "C50812", "C50911", "C50312",
            "C50311", "C50111", "1741", "C5091", "C50811", "1748",
            "C50511", "1743", "C50211", "C50011", "C5051", "C50012",
            "C50419", "1742", "C50611", "C50612", "C50119", "C50819",
            "1746", "C5041", "C50619", "19881", "C5081", "1745",
            "C50219", "C50319", "C50019", "C50519", "C50929", "C50021",
            "C5021", "C5011", "C5031", "C509", "C50", "1759", "C5001",
            "C50421", "C50922", "C50921"
        ]
    )

    st.markdown("""
        <p class="note">Las variables sociodemográficas se obtienen
        automáticamente a partir del ZIP3 seleccionado.</p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# BOTÓN DE PREDICCIÓN
# =========================================================

st.write("")
predecir = st.button("REALIZAR PREDICCIÓN", use_container_width=True)

if predecir:

    datos_zip = zip_data[zip_data["patient_zip3"] == zip3]

    if datos_zip.empty:
        st.error(f"El ZIP3 {zip3} no se encuentra en la base de datos.")
        st.stop()

    datos_zip = datos_zip.iloc[0]

    datos = [[
        Payer,
        Edad,
        Diagnostico,
        datos_zip["population"],
        datos_zip["density"],
        datos_zip["age_median"],
        datos_zip["age_60s"],
        datos_zip["age_70s"],
        datos_zip["married"],
        datos_zip["never_married"],
        datos_zip["family_size"],
        datos_zip["income_household_median"],
        datos_zip["income_household_10_to_15"],
        datos_zip["income_household_75_to_100"],
        datos_zip["income_household_100_to_150"],
        datos_zip["income_household_150_over"],
        datos_zip["income_household_six_figure"],
        datos_zip["income_individual_median"],
        datos_zip["home_ownership"],
        datos_zip["housing_units"],
        datos_zip["rent_median"],
        datos_zip["rent_burden"],
        datos_zip["education_less_highschool"],
        datos_zip["education_highschool"],
        datos_zip["education_bachelors"],
        datos_zip["education_college_or_above"],
        datos_zip["labor_force_participation"],
        datos_zip["unemployment_rate"],
        datos_zip["self_employed"],
        datos_zip["farmer"],
        datos_zip["race_white"],
        datos_zip["race_other"],
        datos_zip["race_multiple"],
        datos_zip["hispanic"],
        datos_zip["disabled"],
        datos_zip["poverty"],
        datos_zip["limited_english"],
        datos_zip["veteran"],
        datos_zip["Ozone"]
    ]]

    columnas = [
        "payer_type", "patient_age", "breast_cancer_diagnosis_code",
        "population", "density", "age_median", "age_60s", "age_70s",
        "married", "never_married", "family_size",
        "income_household_median", "income_household_10_to_15",
        "income_household_75_to_100", "income_household_100_to_150",
        "income_household_150_over", "income_household_six_figure",
        "income_individual_median", "home_ownership", "housing_units",
        "rent_median", "rent_burden", "education_less_highschool",
        "education_highschool", "education_bachelors",
        "education_college_or_above", "labor_force_participation",
        "unemployment_rate", "self_employed", "farmer", "race_white",
        "race_other", "race_multiple", "hispanic", "disabled", "poverty",
        "limited_english", "veteran", "Ozone"
    ]

    data = pd.DataFrame(datos, columns=columnas)

    Y_pred = modelo.predict(data)
    resultado = label_encoder.inverse_transform(Y_pred)

    st.success("Predicción realizada correctamente. Recuerda que el modelo presenta un porcentaje de error aproximado del 56 %.")

    st.markdown(f"""
    <div class="resultado-box">
        <div class="label">Resultado de la predicción</div>
        <div class="valor">{resultado[0]}</div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- Datos utilizados ----------------
    st.markdown('<div class="card"><h3>Datos utilizados</h3>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("ZIP3", zip3)
    with c2:
        st.metric("Edad", Edad)
    with c3:
        st.metric("Pagador", Payer)
    with c4:
        st.metric("Diagnóstico", Diagnostico)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Info sociodemográfica (ampliada) ----------------

    def fmt(valor, tipo="num"):
        """Formatea un valor del ZIP3, controlando valores faltantes."""
        if pd.isna(valor):
            return "No disponible"
        if tipo == "moneda":
            return f"${valor:,.0f}"
        if tipo == "pct":
            return f"{valor:.1f}%"
        if tipo == "dec":
            return f"{valor:,.2f}"
        return f"{valor:,.0f}"

    st.markdown('<div class="card"><h3>Información sociodemográfica del ZIP3</h3>', unsafe_allow_html=True)

    # --- Demografía general ---
    st.markdown('<div class="group-title first">Demografía general</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Población", fmt(datos_zip["population"]))
    with c2:
        st.metric("Densidad", fmt(datos_zip["density"], "dec"))
    with c3:
        st.metric("Edad mediana", fmt(datos_zip["age_median"], "dec"))
    with c4:
        st.metric("Tamaño familiar", fmt(datos_zip["family_size"], "dec"))

    # --- Hogar y vivienda ---
    st.markdown('<div class="group-title">Hogar y vivienda</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Casados", fmt(datos_zip["married"], "pct"))
    with c2:
        st.metric("Propiedad de vivienda", fmt(datos_zip["home_ownership"], "pct"))
    with c3:
        st.metric("Renta mediana", fmt(datos_zip["rent_median"], "moneda"))
    with c4:
        st.metric("Carga de renta", fmt(datos_zip["rent_burden"], "pct"))

    # --- Ingresos ---
    st.markdown('<div class="group-title">Ingresos</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Ingreso hogar (mediana)", fmt(datos_zip["income_household_median"], "moneda"))
    with c2:
        st.metric("Ingreso individual (mediana)", fmt(datos_zip["income_individual_median"], "moneda"))
    with c3:
        st.metric("Hogares seis cifras", fmt(datos_zip["income_household_six_figure"], "pct"))
    with c4:
        st.metric("Pobreza", fmt(datos_zip["poverty"], "pct"))

    # --- Educación y empleo ---
    st.markdown('<div class="group-title">Educación y empleo</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Educación universitaria+", fmt(datos_zip["education_college_or_above"], "pct"))
    with c2:
        st.metric("Bachillerato completo", fmt(datos_zip["education_highschool"], "pct"))
    with c3:
        st.metric("Participación laboral", fmt(datos_zip["labor_force_participation"], "pct"))
    with c4:
        st.metric("Desempleo", fmt(datos_zip["unemployment_rate"], "pct"))

    # --- Composición poblacional ---
    st.markdown('<div class="group-title">Composición poblacional</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Población blanca", fmt(datos_zip["race_white"], "pct"))
    with c2:
        st.metric("Población hispana", fmt(datos_zip["hispanic"], "pct"))
    with c3:
        st.metric("Personas con discapacidad", fmt(datos_zip["disabled"], "pct"))
    with c4:
        st.metric("Veteranos", fmt(datos_zip["veteran"], "pct"))

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # cierre .lift


# =========================================================
# PIE DE PÁGINA
# =========================================================

st.divider()
st.markdown(
    '<p class="footer-note">MediPredict · Sistema académico de predicción '
    'basado en Machine Learning · Random Forest</p>',
    unsafe_allow_html=True
)
