from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 1. Configuración de la Página
st.set_page_config(
    page_title="DULCINEA - Monitor Cohorte VIH Antioquia",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Funciones de Carga de Datos con Caché
BASE_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
DIR_SILVER = BASE_DIR / "data" / "silver"
DIR_GOLD = BASE_DIR / "data" / "gold"


@st.cache_data
def cargar_datos_silver():
    ruta_parquet = DIR_SILVER / "cohorte_vih_silver.parquet"
    if ruta_parquet.exists():
        df = pd.read_parquet(ruta_parquet)
    else:
        ruta_csv = DIR_SILVER / "cohorte_vih_silver.csv"
        df = pd.read_csv(ruta_csv)

    # Traer la probabilidad calculada por el modelo entrenado en 03_Gold_Analytics.ipynb
    # (antes la app recalculaba un riesgo distinto con su propia fórmula, que nunca
    # coincidía con el modelo real -> esto los deja consistentes)
    ruta_pred = DIR_GOLD / "prediccion_riesgo_abandono.csv"
    if ruta_pred.exists():
        df_pred = pd.read_csv(
            ruta_pred, usecols=["Numero de Identificacion", "Probabilidad_Abandono_%", "Nivel_Riesgo_Estimado"]
        )
        df["Numero de Identificacion"] = df["Numero de Identificacion"].astype(str)
        df_pred["Numero de Identificacion"] = df_pred["Numero de Identificacion"].astype(str)
        df = df.merge(df_pred, on="Numero de Identificacion", how="left")

    # Respaldo si aún no se ha corrido el notebook de Gold con el modelo
    if "Probabilidad_Abandono_%" not in df.columns:
        cd4 = df["75.1 Resultado del último conteo de linfocitos TCD4"].fillna(700)
        semanas = df["Semanas en TAR"].fillna(50)
        dias_tar = df["Días oportunidad inicio TAR"].fillna(24)
        z_score = (
            0.5
            - ((cd4 - 700) / 300) * 0.8
            - ((semanas - 50) / 40) * 1.5
            + ((dias_tar - 20) / 10) * 0.6
        )
        df["Probabilidad_Abandono_%"] = ((1 / (1 + np.exp(-z_score))) * 100).round(2)
        df["Nivel_Riesgo_Estimado"] = pd.cut(
            df["Probabilidad_Abandono_%"],
            bins=[-1, 35, 65, 100],
            labels=["Riesgo Bajo", "Riesgo Medio", "Riesgo Alto"],
        )
    return df


df = cargar_datos_silver()

# 3. Encabezado y Título Principal
st.title(" Proyecto DULCINEA: Monitor de Adherencia y Abandono en TAR")
st.markdown(
    "**Gobernación de Antioquia — Secretaría Seccional de Salud y Protección Social**  \n"
    "*Articulación de Datos SIVIGILA, Cuenta de Alto Costo (CAC) y Cohorte VIH. Responsable Técnica: Claudia C.*"
)
st.markdown("---")

# 4. Filtros Laterales Interactivos
st.sidebar.header("🔍 Filtros Operativos")
subregiones_opt = sorted(df["Subregion"].dropna().unique().tolist())
subregiones_sel = st.sidebar.multiselect(
    "Subregión de Residencia:",
    options=subregiones_opt,
    default=subregiones_opt,
)

df_filtrado = df[df["Subregion"].isin(subregiones_sel)]

municipios_opt = sorted(df_filtrado["Municipio residencia"].dropna().unique().tolist())
municipios_sel = st.sidebar.multiselect(
    "Municipio de Residencia:",
    options=municipios_opt,
    default=municipios_opt,
)

df_filtrado = df_filtrado[df_filtrado["Municipio residencia"].isin(municipios_sel)]

estados_opt = df_filtrado["Estado_Seguimiento_VIH"].unique().tolist()
estados_sel = st.sidebar.multiselect(
    "Estado de Seguimiento Clínico:",
    options=estados_opt,
    default=estados_opt,
)

df_filtrado = df_filtrado[df_filtrado["Estado_Seguimiento_VIH"].isin(estados_sel)]

st.sidebar.markdown("---")
st.sidebar.caption("📊 Base Activa: **850 Registros Únicos** (Deduplicados)")

# 5. Tarjetas de KPIs Principales
col1, col2, col3, col4, col5 = st.columns(5)

total_p = len(df_filtrado)
n_abandono = len(
    df_filtrado[df_filtrado["Estado_Seguimiento_VIH"] == "Abandono / Sin TAR"]
)
n_control = len(
    df_filtrado[
        df_filtrado["Estado_Seguimiento_VIH"] == "En TAR - Virológicamente Controlado"
    ]
)
n_falla = len(
    df_filtrado[
        df_filtrado["Estado_Seguimiento_VIH"] == "En TAR - Con Carga Viral Detectable"
    ]
)

tasa_abandono = (n_abandono / total_p * 100) if total_p > 0 else 0
promedio_cd4 = df_filtrado["75.1 Resultado del último conteo de linfocitos TCD4"].mean()

col1.metric("Cohorte Filtrada", f"{total_p:,} pac.")
col2.metric("Abandono / Sin TAR", f"{n_abandono:,}", delta_color="inverse")
col3.metric("En Falla Viral", f"{n_falla:,}")
col4.metric("Tasa de Abandono", f"{tasa_abandono:.1f}%")
col5.metric("Promedio CD4", f"{promedio_cd4:.0f} cel/mm³")

st.markdown("---")

# 6. Pestañas Principales del Dashboard
tab1, tab2, tab3 = st.tabs(
    [
        "📈 Monitoreo Epidemiológico",
        "🔮 Modelo Predictivo de Riesgo",
        "📋 Gestión Operativa y Campo",
    ]
)

# -----------------------------------------------------------------------------
# TAB 1: MONITOREO EPIDEMIOLÓGICO Y TENDENCIA CENTRAL
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Estado del Tratamiento Antirretroviral (TAR) y Distribución Territorial")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("**Distribución de la Cohorte por Estado de Seguimiento**")
        fig_pie = px.pie(
            df_filtrado,
            names="Estado_Seguimiento_VIH",
            color="Estado_Seguimiento_VIH",
            color_discrete_map={
                "En TAR - Virológicamente Controlado": "#2ecc71",
                "En TAR - Con Carga Viral Detectable": "#f39c12",
                "Abandono / Sin TAR": "#e74c3c",
            },
            hole=0.4,
        )
        fig_pie.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_chart2:
        st.markdown("**Pacientes en Abandono por Municipio de Residencia**")
        df_abandono_mun = (
            df_filtrado[df_filtrado["Estado_Seguimiento_VIH"] == "Abandono / Sin TAR"]
            .groupby("Municipio residencia")
            .size()
            .reset_index(name="Casos_Abandono")
            .sort_values(by="Casos_Abandono", ascending=True)
        )
        fig_bar = px.bar(
            df_abandono_mun,
            y="Municipio residencia",
            x="Casos_Abandono",
            orientation="h",
            color="Casos_Abandono",
            color_continuous_scale="Reds",
            text="Casos_Abandono",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("### Medidas de Tendencia Central y Dispersión (Página 2 del Requerimiento)")
    
    col_cd4_var = "75.1 Resultado del último conteo de linfocitos TCD4"
    col_dias_var = "Días oportunidad inicio TAR"
    
    stats_df = pd.DataFrame(
        {
            "Variable Cuantitativa": [
                "Edad del Paciente (Años)",
                "Conteo de Linfocitos CD4 (cel/mm³)",
                "Días de Oportunidad Inicio TAR",
            ],
            "Promedio (Media)": [
                df_filtrado["Edad"].mean(),
                df_filtrado[col_cd4_var].mean(),
                df_filtrado[col_dias_var].mean(),
            ],
            "Mediana": [
                df_filtrado["Edad"].median(),
                df_filtrado[col_cd4_var].median(),
                df_filtrado[col_dias_var].median(),
            ],
            "Desviación Estándar": [
                df_filtrado["Edad"].std(),
                df_filtrado[col_cd4_var].std(),
                df_filtrado[col_dias_var].std(),
            ],
            "Mínimo": [
                df_filtrado["Edad"].min(),
                df_filtrado[col_cd4_var].min(),
                df_filtrado[col_dias_var].min(),
            ],
            "Máximo": [
                df_filtrado["Edad"].max(),
                df_filtrado[col_cd4_var].max(),
                df_filtrado[col_dias_var].max(),
            ],
        }
    ).round(2)
    
    st.table(stats_df)

# -----------------------------------------------------------------------------
# TAB 2: MODELO PREDICTIVO DE RIESGO
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Estimación Algorítmica del Riesgo de Deserción y Falla Virológica")
    
    col_pred1, col_pred2 = st.columns([1, 1])
    
    with col_pred1:
        st.markdown("**Estratificación Epidemiológica de Riesgo (Total Cohorte)**")
        fig_risk = px.histogram(
            df_filtrado,
            x="Nivel_Riesgo_Estimado",
            color="Nivel_Riesgo_Estimado",
            color_discrete_map={
                "Riesgo Bajo": "#2ecc71",
                "Riesgo Medio": "#f39c12",
                "Riesgo Alto": "#e74c3c",
            },
            text_auto=True,
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
    with col_pred2:
        st.markdown("**Simulador de Riesgo Individual para Atención Primaria**")
        st.caption("Ajuste los parámetros clínicos para estimar la probabilidad de desafección del paciente:")
        
        sim_recibe_tar = st.radio("¿El paciente recibe TAR actualmente?", ["Sí", "No"], horizontal=True)
        sim_cd4 = st.slider("Conteo de Linfocitos CD4 (cel/mm³)", 100, 1200, 450)
        sim_semanas = st.slider("Semanas Transcurridas en TAR", 0, 200, 24)
        sim_dias = st.slider("Días Transcurridos hasta Inicio TAR", 1, 60, 20)

        # Cálculo dinámico (paciente sin TAR se trata como riesgo alto directo,
        # para no confundir "recién inició TAR" con "no está en TAR")
        if sim_recibe_tar == "No":
            prob_sim = 90.0
        else:
            z_sim = 0.5 - ((sim_cd4 - 700) / 300) * 0.8 - ((sim_semanas - 50) / 40) * 1.5 + ((sim_dias - 20) / 10) * 0.6
            prob_sim = (1 / (1 + np.exp(-z_sim))) * 100
        
        st.markdown(f"#### Probabilidad Estimada de Abandono: **{prob_sim:.1f}%**")
        
        if prob_sim > 65:
            st.error("🚨 **RIESGO ALTO:** Paciente prioritario para intervención de enfermería y seguimiento telefónico.")
        elif prob_sim > 35:
            st.warning("⚠️ **RIESGO MEDIO:** Paciente requiere refuerzo en adherencia y agenda de control próximo.")
        else:
            st.success("✅ **RIESGO BAJO:** Paciente adherente. Mantener controles de rutina.")

# -----------------------------------------------------------------------------
# TAB 3: GESTIÓN OPERATIVA Y BÚSQUEDA ACTIVA EN CAMPO
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Búsqueda Activa de Pacientes para Médicos e IPS Primarias")
    
    filtro_estado = st.radio(
        "Mostrar Pacientes en:",
        options=["Solo Abandono / Sin TAR", "Riesgo Alto Estimado", "Toda la Cohorte"],
        horizontal=True,
    )
    
    if filtro_estado == "Solo Abandono / Sin TAR":
        df_tabla = df_filtrado[df_filtrado["Estado_Seguimiento_VIH"] == "Abandono / Sin TAR"]
    elif filtro_estado == "Riesgo Alto Estimado":
        df_tabla = df_filtrado[df_filtrado["Nivel_Riesgo_Estimado"] == "Riesgo Alto"]
    else:
        df_tabla = df_filtrado
        
    cols_mostrar = [
        "TD",
        "Numero de Identificacion",
        "Nombre_Completo",
        "Municipio residencia",
        "Subregion",
        "Edad",
        "Estado_Seguimiento_VIH",
        "Probabilidad_Abandono_%",
        "Nivel_Riesgo_Estimado",
        "Nombre IPS/ESE primaria",
    ]
    
    cols_existentes = [c for c in cols_mostrar if c in df_tabla.columns]
    
    st.dataframe(df_tabla[cols_existentes], use_container_width=True, hide_index=True)
    
    # Botón de Descarga para Búsqueda Activa
    csv_data = df_tabla[cols_existentes].to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 Descargar Listado de Pacientes para Búsqueda Activa (CSV)",
        data=csv_data,
        file_name="Listado_Pacientes_DULCINEA_Campo.csv",
        mime="text/csv",
    )