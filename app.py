import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- VARIABLES CLAVE PARA SIMULACIÓN ---
# 🚨 IMPORTANTE: Revisa el contenido de tu 'selected_vars_volatilidad.pkl'.
# Los nombres deben coincidir EXACTAMENTE (mayúsculas/minúsculas) con los de tu CSV.
# Ejemplo, si tu CSV tiene 'precio_cobre' en minúsculas, úsalo aquí.
KEY_SIMULATION_VARS = [
    "precio_cobre",
    "reservas",
    "Tasa_Referencia",
    # Añade o ajusta los nombres de las variables macroeconómicas más importantes
]
# ---------------------------------------

sns.set(style="whitegrid")

st.set_page_config(
    page_title="Volatilidad del Tipo de Cambio",
    layout="wide"
)

# --------------------------------------------------------------------
# CONSTANTES: TIMELINE + MAPA DE MESES (Sin cambios, son constantes de la interfaz)
# --------------------------------------------------------------------
IMAGES = [
    "https://raw.githubusercontent.com/LuisCamposD/timeline_s1/main/timeline_images/img1.jpg",
    "https://raw.githubusercontent.com/LuisCamposD/timeline_s1/main/timeline_images/img2.PNG",
    "https://raw.githubusercontent.com/LuisCamposD/timeline_s1/main/timeline_images/img3.jpg",
    "https://raw.githubusercontent.com/LuisCamposD/timeline_s1/main/timeline_images/img4.png",
    "https://raw.githubusercontent.com/LuisCamposD/timeline_s1/main/timeline_images/img5.png",
]

CAPTIONS = [
    "Años 80–90: enfoque básico",
    "Años 2000: apertura comercial y mayor exposición al dólar",
    "2008–2012: crisis y gestión del riesgo",
    "2013–2019: digitalización, BI y monitoreo diario del tipo de cambio",
    "2020 en adelante: disrupciones globales, analítica avanzada e IA",
]

TIMELINE = [
    {
        "titulo": "1️⃣ Años 80–90: tipo de cambio y compras casi desconectados",
        "resumen": ("En esta etapa el análisis de la volatilidad era mínimo. El tipo de cambio se veía como un dato macro, no como un insumo clave para las decisiones de logística."),
        "bullets": ["Planeación de compras principalmente basada en experiencia y listas de precios históricas.", "Poca apertura comercial: menor participación de proveedores internacionales.", "El tipo de cambio se revisaba esporádicamente, no todos los días.", "No existían políticas claras sobre quién asumía el riesgo cambiario (proveedor vs empresa)."],
    },
    {
        "titulo": "2️⃣ Años 2000: apertura comercial y mayor exposición al dólar",
        "resumen": ("Con la globalización y el aumento de importaciones, el tipo de cambio empieza a impactar directamente los costos logísticos."),
        "bullets": ["Más compras en dólares (equipos, repuestos, tecnología, mobiliario importado).", "Compras empieza a comparar cotizaciones en distintas monedas, pero el análisis es manual (Excel básico).", "Se empiezan a usar tipos de cambio referenciales para presupuestos, pero sin escenarios de volatilidad.", "Mayor sensibilidad en los márgenes: variaciones de centavos ya impactan el costo total de los proyectos."],
    },
    {
        "titulo": "3️⃣ 2008–2012: crisis financiera y prioridad al riesgo cambiario",
        "resumen": ("La crisis global y los saltos bruscos del tipo de cambio obligan a formalizar la gestión del riesgo cambiario en compras y contratos."),
        "bullets": ["Logística y Finanzas comienzan a trabajar juntos para definir TC de referencia y bandas de variación.", "Aparecen cláusulas específicas: ajuste de precio por tipo de cambio, vigencia corta de cotizaciones.", "Se analizan escenarios básicos: ¿qué pasa si el dólar sube 5%, 10% durante el proyecto?", "Compras prioriza cerrar rápidamente órdenes de compra críticas para evitar descalce entre aprobación y pago."],
    },
    {
        "titulo": "4️⃣ 2013–2019: digitalización, BI y monitoreo diario del tipo de cambio",
        "resumen": ("Las empresas adoptan ERPs, dashboards y reportes automáticos. El tipo de cambio se vuelve un indicador operativo para logística."),
        "bullets": ["Dashboards de compras que muestran el impacto del tipo de cambio en el presupuesto y en el costo por contrato.", "Actualización diaria del tipo de cambio en sistemas (ERP) y en las plantillas de cuadros comparativos.", "Uso de modelos estadísticos simples para proyectar TC anual y armar presupuestos más realistas.", "Compras empieza a definir estrategias: adelantar o postergar compras según tendencias de tipo de cambio."],
    },
    {
        "titulo": "5️⃣ 2020 en adelante: disrupciones globales, analítica avanzada e IA",
        "resumen": ("Con la pandemia y los choques globales, la volatilidad del tipo de cambio se combina con rupturas de cadena de suministro. Compras necesita decisiones más inteligentes y rápidas."),
        "bullets": ["Uso de analítica avanzada e IA para simular escenarios de tipo de cambio y su efecto en costos logísticos.", "Modelos que recomiendan: comprar ahora vs esperar, cambiar de proveedor, negociar en otra moneda o ajustar incoterms.", "Integración de datos de mercado (TC, commodities, fletes internacionales) con datos internos de consumo y stock.", "El rol de Compras/Logística evoluciona: de ejecutor de órdenes a gestor estratégico del riesgo cambiario y de suministro."],
    },
]

MAPA_MESES = {
    "Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Ago": 8, "Set": 9, "Sep": 9, "Oct": 10, "Nov": 11, "Dic": 12,
}

# ---------- 1. Cargar modelo, imputer, scaler, variables y datos ----------
@st.cache_resource
def cargar_recursos():
    try:
        modelo = joblib.load("gbr_mejor_modelo_tc.pkl")
        selected_vars = joblib.load("selected_vars_volatilidad.pkl")
        imputer = joblib.load("imputer_volatilidad.pkl")
        scaler = joblib.load("scaler_volatilidad.pkl")
        df = pd.read_csv("datos_tc_limpios.csv")
    except FileNotFoundError as e:
        st.error(f"Archivo necesario no encontrado: {e.filename}. Asegúrate de que todos los .pkl y el .csv estén en la raíz de tu GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar recursos: {e}")
        st.exception(e) # Mostrar la excepción completa en los logs
        st.stop()


    # Detectar columna TC
    posibles_tc = [
        "TC", "tc", "TC_venta", "tc_venta",
        "Tipo de cambio - TC Sistema bancario SBS (S/ por US$) - Venta",
        "Tipo de cambio - TC Sistema bancario SBS (S/ por US$) - Venta ",
        "Tipo_de_cambio",
    ]

    tc_col = None
    for col in posibles_tc:
        if col in df.columns:
            tc_col = col
            break

    if tc_col is None:
        for col in df.columns:
            nombre = col.lower()
            if "tipo de cambio" in nombre or nombre == "tc":
                tc_col = col
                break

    if tc_col is None:
        # Si la columna TC no se encuentra (el problema que ya encontramos y resolvimos),
        # lanzamos un error claro.
        raise KeyError(
            f"No se encontró la columna de Tipo de Cambio en el CSV. "
            f"Columna TC usada: {tc_col}. Columnas disponibles: {list(df.columns)}"
        )

    # Crear fecha y ordenar
    if "fecha" not in df.columns:
        if "anio" in df.columns and "mes" in df.columns:
            df["mes"] = df["mes"].astype(str)
            df["mes_num"] = df["mes"].map(MAPA_MESES)
            df["fecha"] = pd.to_datetime(
                dict(year=df["anio"], month=df["mes_num"], day=1)
            )
        else:
            df["fecha"] = pd.date_range(start="2000-01-01", periods=len(df), freq="M")
    else:
        df["fecha"] = pd.to_datetime(df["fecha"])

    if "mes_num" not in df.columns and "mes" in df.columns:
        df["mes"] = df["mes"].astype(str)
        df["mes_num"] = df["mes"].map(MAPA_MESES)

    df = df.sort_values("fecha").reset_index(drop=True)

    # Rendimientos logarítmicos
    df_mod = df.copy()
    
    # LÍNEA DE SEGURIDAD: Convertir columnas numéricas a float antes del cálculo
    for col in df_mod.columns:
        if col not in ["fecha", "mes"]: 
            df_mod[col] = pd.to_numeric(df_mod[col], errors='coerce') 

    df_mod["Rendimientos_log"] = np.log(df_mod[tc_col] / df_mod[tc_col].shift(1))
    df_mod = df_mod.dropna(subset=["Rendimientos_log"])

    return modelo, imputer, scaler, selected_vars, df, df_mod, tc_col


st.write("🔄 Inicializando app y cargando recursos...")

try:
    modelo, imputer, scaler, selected_vars, df, df_mod, tc_col = cargar_recursos()
except Exception as e:
    st.error("❌ Error cargando los recursos (modelo, datos o transformaciones). Por favor, revise los logs.")
    st.exception(e)
    st.stop()


# --------------------------------------------------------------------
# Extracción de valores base para la simulación
# --------------------------------------------------------------------
df_ordenado = df.sort_values("fecha").reset_index(drop=True)
ultimo_X_base = df_mod.iloc[-1].copy() # Cambiado a iloc[-1] para capturar todos los features en el índice correcto
ultimo_tc_base = df_ordenado[tc_col].iloc[-1]

# --------------------------------------------------------------------
# Sidebar: navegación
# --------------------------------------------------------------------
st.sidebar.title("Menú")
pagina = st.sidebar.radio(
    "Ir a:",
    ["Inicio y línea de tiempo", "EDA", "Modelo y predicciones"]
)

# ---------- PÁGINAS (CONTENIDO OMITIDO POR SER ESTÁTICO) ----------
# ...

# --------------------------------------------------------------------
# Página: Modelo y predicciones
# --------------------------------------------------------------------
elif pagina == "Modelo y predicciones":
    st.title("Modelo de Volatilidad y Predicciones")

    # 5.1 Performance del modelo
    st.subheader("Performance del modelo (Evaluación Histórica)")

    X = df_mod # Contiene todas las columnas necesarias
    y = df_mod["Rendimientos_log"]

    train_size = int(len(X) * 0.8)
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]
    
    # -------------------------------------------------------------------
    # FIX: Asegurar el orden de las columnas y el tipo de dato (float) para SKLEARN
    # -------------------------------------------------------------------
    X_test_correct_order = X_test[selected_vars] # Asegura orden
    X_test_data = X_test_correct_order.values.astype(float) # Forzar a float array
    X_test_imp = imputer.transform(X_test_data) # <--- La línea que causaba el error con tipo de dato
    
    X_test_scaled = scaler.transform(X_test_imp)
    y_pred_test = modelo.predict(X_test_scaled)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    # Evaluación en todo el histórico (in-sample)
    X_all_correct_order = X[selected_vars]
    X_all_data = X_all_correct_order.values.astype(float) 
    X_all_imp = imputer.transform(X_all_data)
    
    X_all_scaled = scaler.transform(X_all_imp)
    y_pred_all = modelo.predict(X_all_scaled)

    mae_all = mean_absolute_error(y, y_pred_all)
    rmse_all = np.sqrt(mean_squared_error(y, y_pred_all))
    r2_all = r2_score(y, y_pred_all)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Conjunto de prueba (20% final de la serie)**")
        st.metric("R2 (test)", f"{r2_test:.4f}")
        st.metric("MAE (test)", f"{mae_test:.6f}")
        st.metric("RMSE (test)", f"{rmse_test:.6f}")

    with col2:
        st.markdown("**Todo el histórico (in-sample)**")
        st.metric("R2 (in-sample)", f"{r2_all:.4f}")
        st.metric("MAE (in-sample)", f"{mae_all:.6f}")
        st.metric("RMSE (in-sample)", f"{rmse_all:.6f}")

    st.markdown("### Rendimientos logarítmicos en el conjunto de prueba")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(y_test.values, label="Real", alpha=0.8)
    ax.plot(y_pred_test, label="Predicho", alpha=0.8)
    ax.set_title("Rendimientos logarítmicos: real vs predicho (test)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # 5.2 Predicción futura con simulación
    st.markdown("---")
    st.subheader("Predicción de varios meses hacia adelante con Simulación de Variables")

    st.write("""
    **Para obtener una predicción realista, debes simular los valores futuros de las variables clave.**
    Las variables no simuladas se mantendrán en el último valor histórico.
    """)

    col_a, col_b, col_c = st.columns(3)
    
    # ------------------ INPUTS DE TIEMPO ------------------
    ultimo_mes_nombre = df_ordenado["mes"].iloc[-1] if "mes" in df_ordenado.columns else "Dic"
    ultimo_anio = int(df_ordenado["anio"].iloc[-1]) if "anio" in df_ordenado.columns else pd.Timestamp.today().year

    meses_nombres = list(MAPA_MESES.keys())
    try:
        idx_mes_default = meses_nombres.index(ultimo_mes_nombre)
    except ValueError:
        idx_mes_default = 11

    with col_a:
        anio_input = st.number_input(
            "Año de inicio de la predicción (normalmente el último año)",
            min_value=df_ordenado["anio"].min(),
            max_value=ultimo_anio + 10,
            value=ultimo_anio,
            step=1,
            key="anio_pred"
        )
    with col_b:
        mes_nombre = st.selectbox(
            "Mes de inicio (el mes siguiente al último dato)",
            options=meses_nombres,
            index=idx_mes_default,
            key="mes_pred"
        )
        mes_inicio = MAPA_MESES[mes_nombre]
    with col_c:
        num_meses = st.slider("Número de meses a predecir", 1, 24, 6)

    # ------------------ INPUTS DE SIMULACIÓN ------------------
    st.markdown("#### Valores de Simulación para el Período Futuro")
    
    # Filtramos KEY_SIMULATION_VARS a solo las que estén realmente en selected_vars
    sim_vars_actual = [var for var in KEY_SIMULATION_VARS if var in ultimo_X_base.index and var in selected_vars]
    
    simulated_values = {}
    
    # 🚨 FIX CRÍTICO: Control de flujo para evitar st.columns(0)
    if len(sim_vars_actual) > 0:
        
        # Crear columnas solo si hay variables que simular
        cols_sim = st.columns(len(sim_vars_actual))
        
        # Generar Sliders para variables clave
        for i, var in enumerate(sim_vars_actual):
            last_value = ultimo_X_base[var]
            
            # Definir un rango razonable basado en el último valor
            min_val = last_value * 0.9 if last_value > 0 else last_value - abs(last_value * 0.1)
            max_val = last_value * 1.1 if last_value > 0 else last_val + abs(last_val * 0.1)
            
            # Usar 4 decimales si el valor es muy pequeño
            step_val = 0.0001 if abs(last_value) < 1.0 else 0.01

            with cols_sim[i]:
                simulated_values[var] = st.slider(
                    f"Valor de {var} (último: {last_value:.4f})",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(last_value),
                    step=step_val,
                    format="%.4f",
                    key=f"sim_{var}"
                )
    else:
        st.warning(
            "⚠️ Ninguna variable clave fue encontrada. Se usarán los últimos valores históricos."
        )


    if st.button("Calcular predicción", key="btn_prediccion"):
        
        with st.spinner('Generando la proyección con los escenarios simulados...'):
            
            # Generar meses futuros
            meses_futuro = []
            mes_actual = mes_inicio
            anio_actual = int(anio_input)

            if anio_actual == ultimo_anio and mes_actual == MAPA_MESES.get(ultimo_mes_nombre):
                mes_actual += 1
                if mes_actual > 12:
                    mes_actual = 1
                    anio_actual += 1
            
            for _ in range(num_meses):
                meses_futuro.append((anio_actual, mes_actual))
                mes_actual += 1
                if mes_actual > 12:
                    mes_actual = 1
                    anio_actual += 1
            
            df_futuro = pd.DataFrame(meses_futuro, columns=["anio", "mes_num"])

            # -----------------------------------------------------------
            # LÓGICA DE ASIGNACIÓN DE FEATURES FUTUROS
            # -----------------------------------------------------------
            for col in selected_vars:
                if col in ["anio", "mes_num"]:
                    continue
                
                # Asigna valor simulado si existe, si no, usa el último valor histórico (ultimo_X_base)
                if col in simulated_values:
                    df_futuro[col] = simulated_values[col]
                else:
                    # Copiamos el último valor conocido si no se está simulando
                    df_futuro[col] = ultimo_X_base[col]


            # -----------------------------------------------------------
            # FIX: Asegurar el orden de las columnas y el tipo de dato (float) para SKLEARN
            # -----------------------------------------------------------
            X_fut_correct_order = df_futuro[selected_vars]
            X_fut_data = X_fut_correct_order.values.astype(float) 
            X_fut_imp = imputer.transform(X_fut_data)
            
            X_fut_scaled = scaler.transform(X_fut_imp)

            # Predicción directa de rendimientos logarítmicos
            rendimientos_pred = modelo.predict(X_fut_scaled)

            # Reconstrucción del tipo de cambio a partir del último TC histórico
            tc_pred = []
            ultimo_tc = ultimo_tc_base
            
            for r in rendimientos_pred:
                nuevo_tc = ultimo_tc * np.exp(r)
                tc_pred.append(nuevo_tc)
                ultimo_tc = nuevo_tc 

            df_futuro["TC_predicho"] = tc_pred

            # Mapear número de mes a nombre
            mes_dict_inv = {v: k for k, v in MAPA_MESES.items()}
            df_futuro["mes"] = df_futuro["mes_num"].map(mes_dict_inv)
            df_futuro["anio"] = df_futuro["anio"].astype(int)

            st.write("### Predicciones futuras")
            
            df_display = df_futuro[["anio", "mes", "TC_predicho"]].copy()
            df_display["TC_predicho"] = df_display["TC_predicho"].apply(lambda x: f"{x:.4f}")
            st.dataframe(df_display, use_container_width=True)

            # -----------------------------------------------------------
            # GRÁFICO FINAL (Histórico + Forecast)
            # -----------------------------------------------------------
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Histórico (Serie Azul)
            x_hist = np.arange(len(df_ordenado))
            ax.plot(x_hist, df_ordenado[tc_col], label="TC real (histórico)", linewidth=2, color='blue')
            
            # Predicción (Serie Roja, Punteada)
            x_fut = np.arange(len(df_ordenado), len(df_ordenado) + num_meses)
            ax.plot(
                x_fut,
                df_futuro["TC_predicho"],
                label="TC predicho (Simulación)",
                marker="o",
                linestyle='--',
                color='red'
            )
            
            # Línea vertical que marca el inicio de la predicción
            ax.axvline(x=len(df_ordenado) - 1, color='gray', linestyle=':', label='Fin del Histórico')

            ax.set_title(
                f"Proyección del Tipo de Cambio - {num_meses} meses a partir de la Simulación"
            )
            ax.set_xlabel("Punto en la Serie Temporal (Meses)")
            ax.set_ylabel("Tipo de cambio (S/ por US$)")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            
            st.success("Predicción completada. Ajusta los parámetros de simulación para ver otros escenarios.")
