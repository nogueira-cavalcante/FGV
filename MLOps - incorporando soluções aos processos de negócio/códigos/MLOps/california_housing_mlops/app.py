from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone
import os
import tempfile
from filelock import FileLock

st.set_page_config(
    page_title="Previsão de Casas - Califórnia",
    page_icon="🏠"
)

st.title("Previsão de preços de casas na Califórnia")
st.caption("Regressão com pipeline versionado no MLflow")

# ============================================================
# 1. Configuração do projeto e do MLflow
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
MLFLOW_DB = PROJECT_DIR / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.resolve().as_posix()}"
)

MODEL_NAME = "CaliforniaHousing"
MODEL_VERSION = "4"
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
MODEL = "xgboost"
HISTORY_PATH = PROJECT_DIR / "historico_inferencias.xlsx"

# ============================================================
# 2. Carregamento do modelo
# ============================================================

@st.cache_resource
def load_model():
    return mlflow.sklearn.load_model(MODEL_URI)


try:
    model = load_model()
except Exception as e:
    st.error("Não foi possível carregar o modelo do MLflow.")
    st.exception(e)
    st.info(f"Banco de dados configurado: {MLFLOW_DB}")
    st.stop()

st.info(
    f"Modelo em uso: **{MODEL_NAME} — algoritmo {MODEL} — versão {MODEL_VERSION}**"
)


# ============================================================
# 3. Formulário de entrada
# ============================================================

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        medinc = st.number_input(
            "MedInc — renda mediana",
            min_value=0.0, value=3.87, step=0.01
        )
        houseage = st.number_input(
            "HouseAge — idade média",
            min_value=1.0, value=28.0, step=1.0
        )
        averooms = st.number_input(
            "AveRooms — cômodos médios",
            min_value=0.1, value=5.43, step=0.01
        )
        avebedrms = st.number_input(
            "AveBedrms — quartos médios",
            min_value=0.1, value=1.10, step=0.01
        )

    with col2:
        population = st.number_input(
            "Population — população",
            min_value=1.0, value=1425.0, step=1.0
        )
        aveoccup = st.number_input(
            "AveOccup — ocupação média",
            min_value=0.1, value=3.07, step=0.01
        )
        latitude = st.number_input(
            "Latitude", value=35.63, format="%.4f"
        )
        longitude = st.number_input(
            "Longitude", value=-119.57, format="%.4f"
        )

    submitted = st.form_submit_button("Estimar preço")


# ============================================================
# 4. Engenharia de atributos, previsão e registro em Excel
# ============================================================

if submitted:
    from datetime import datetime
    from openpyxl import Workbook, load_workbook

    income_category = pd.cut(
        [medinc],
        bins=[-np.inf, 2, 4, 6, np.inf],
        labels=["Baixa", "Média-baixa", "Média-alta", "Alta"]
    )[0]

    model_input = pd.DataFrame([{
        "MedInc": medinc,
        "HouseAge": houseage,
        "AveRooms": averooms,
        "AveBedrms": avebedrms,
        "Population": population,
        "AveOccup": aveoccup,
        "Latitude": latitude,
        "Longitude": longitude,
        "RoomsPerHousehold": averooms / max(aveoccup, 0.1),
        "BedroomsPerRoom": avebedrms / max(averooms, 0.1),
        "PopulationPerHousehold": population / max(aveoccup, 0.1),
        "MedIncLog": np.log1p(medinc),
        "IncomeCategory": str(income_category),
    }])

    try:
        prediction_in_100k_usd = model.predict(model_input)[0]

        # Captura o instante imediatamente após a previsão.
        timestamp = datetime.now().astimezone().isoformat(
            timespec="microseconds"
        )

        predicted_usd = max(0, float(prediction_in_100k_usd) * 100_000)

        st.success(f"Preço estimado: **US$ {predicted_usd:,.0f}**")
        st.caption(
            f"Saída do modelo: {prediction_in_100k_usd:.3f} "
            "(unidade = centenas de milhares de US$)"
        )

    except Exception as e:
        st.error("Erro ao realizar a previsão.")
        st.exception(e)

    else:
        excel_path = PROJECT_DIR / "historico_inferencias.xlsx"

        colunas = [
            "timestamp",
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
            "previsao_usd",
            "modelo",
            "algoritmo",
            "versao_modelo",
        ]

        registro = [
            timestamp,
            medinc,
            houseage,
            averooms,
            avebedrms,
            population,
            aveoccup,
            latitude,
            longitude,
            predicted_usd,
            MODEL_NAME,
            MODEL,
            MODEL_VERSION,
        ]

        try:
            if excel_path.exists():
                workbook = load_workbook(excel_path)
                sheet = workbook.active
            else:
                workbook = Workbook()
                sheet = workbook.active
                sheet.title = "Inferencias"
                sheet.append(colunas)

            sheet.append(registro)
            workbook.save(excel_path)
            workbook.close()

            st.caption(f"Inferência registrada em: {excel_path}")

        except Exception as e:
            st.warning(
                "A previsão foi realizada, mas não foi possível "
                "salvar o histórico em Excel."
            )
            st.exception(e)
