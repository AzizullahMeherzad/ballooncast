import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import pytz
import plotly.graph_objects as go
from datetime import timedelta
from astral import LocationInfo
from astral.sun import sun
from reportlab.platypus import SimpleDocTemplate, Table
import io

st.title("🎈 Balon Uçuş Tahmin Sistemi")

# -------------------------
# LOAD MODEL
# -------------------------

with open("rf_model.pkl","rb") as f:
    rf_model = pickle.load(f)

with open("imputer.pkl","rb") as f:
    imputer = pickle.load(f)

# -------------------------
# LOCATIONS
# -------------------------

locations = {
"goreme": (38.6431,34.8284),
"cavusin": (38.6730,34.8400),
"uchisar": (38.6300,34.8060)
}

# -------------------------
# SUNRISE WINDOW
# -------------------------

city = LocationInfo("Nevsehir","Turkey","Europe/Istanbul",38.6250,34.7122)
tz = pytz.timezone("Europe/Istanbul")

def get_flight_window(date):

    s = sun(city.observer,date=date)
    sunrise = s["sunrise"].astimezone(tz)

    return sunrise - timedelta(minutes=20), sunrise + timedelta(minutes=90)

# -------------------------
# WEATHER API
# -------------------------

def get_weather_for_location(date,lat,lon):

    date_str = date.strftime("%Y-%m-%d")

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&hourly=windspeed_10m,windgusts_10m,precipitation,snowfall,surface_pressure,cloudcover"
        f"&start_date={date_str}&end_date={date_str}"
        "&timezone=Europe/Istanbul"
    )

    r = requests.get(url)
    data = r.json()

    if "hourly" not in data:
        return None

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("Europe/Istanbul")

    return df

# -------------------------
# FEATURE BUILDER
# -------------------------

def build_features_for_date(date):

    start,end = get_flight_window(date)

    all_windows = []

    for name,(lat,lon) in locations.items():

        df_weather = get_weather_for_location(date,lat,lon)

        if df_weather is None:
            continue

        df_window = df_weather[
            (df_weather["time"]>=start) &
            (df_weather["time"]<=end)
        ]

        if not df_window.empty:
            all_windows.append(df_window)

    if len(all_windows)==0:
        return None

    df_all = pd.concat(all_windows)

    features = {

        "wind_mean":df_all["windspeed_10m"].mean(),
        "wind_std":df_all["windspeed_10m"].std(),
        "gust_mean":df_all["windgusts_10m"].mean(),
        "gust_std":df_all["windgusts_10m"].std(),
        "precip_mean":df_all["precipitation"].mean(),
        "snow_mean":df_all["snowfall"].mean(),
        "pressure_mean":df_all["surface_pressure"].mean(),
        "cloud_mean":df_all["cloudcover"].mean()

    }

    return pd.DataFrame([features])

# -------------------------
# PREDICT DATE
# -------------------------

def predict_date_rf(date):

    X_new = build_features_for_date(date)

    if X_new is None:
        return None

    X_imp = imputer.transform(X_new)

    prob = rf_model.predict_proba(X_imp)[0][1]

    return prob

# -------------------------
# PREDICT MONTH
# -------------------------

def predict_month_rf(year,month):

    dates = pd.date_range(
        f"{year}-{month:02d}-01",
        f"{year}-{month:02d}-{pd.Period(f'{year}-{month:02d}').days_in_month}"
    )

    results = []

    for d in dates:

        prob = predict_date_rf(d)

        if prob is None:
            continue

        if prob < 0.3:
            risk = "High Risk"
        elif prob < 0.6:
            risk = "Medium Risk"
        else:
            risk = "Safe"

        results.append({
            "date":d.date(),
            "probability_%":round(prob*100,1),
            "risk_level":risk
        })

    return pd.DataFrame(results)

# -------------------------
# GAUGE
# -------------------------

def show_gauge(prob):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={'text':"Flight Probability"},
        gauge={
            'axis':{'range':[0,100]},
            'steps':[
                {'range':[0,40],'color':"red"},
                {'range':[40,60],'color':"yellow"},
                {'range':[60,100],'color':"green"}
            ]
        }
    ))

    return fig

# -------------------------
# PDF EXPORT
# -------------------------

def create_pdf(df):

    buffer = io.BytesIO()

    table_data = [df.columns.tolist()] + df.values.tolist()

    pdf = SimpleDocTemplate(buffer)

    table = Table(table_data)

    pdf.build([table])

    buffer.seek(0)

    return buffer

# -------------------------
# MENU
# -------------------------

menu = st.sidebar.selectbox(
"Menü",
["Günlük Tahmin","Aylık Tahmin","Model Performansı"]
)

# -------------------------
# DAILY UI
# -------------------------

if menu == "Günlük Tahmin":

    date = st.date_input("Tarih Seç")

    if st.button("Tahmin Yap"):

        prob = predict_date_rf(pd.to_datetime(date))

        if prob is None:

            st.error("Weather data unavailable")

        else:

            st.subheader(f"Uçuş Olasılığı: {round(prob*100,1)}%")

            fig = show_gauge(prob)

            st.plotly_chart(fig)

            if prob >= 0.6:
                st.success("🟢 Uçuş İçin Güvenli")
            elif prob >= 0.4:
                st.warning("🟡 Orta Risk")
            else:
                st.error("🔴 Yüksek Risk")

# -------------------------
# MONTH UI
# -------------------------

elif menu == "Aylık Tahmin":

    year = st.number_input("Year",value=2026)
    month = st.number_input("Month",min_value=1,max_value=12,value=3)

    if st.button("Aylık Tahmin"):

        df = predict_month_rf(year,month)

        st.dataframe(df)

        pdf = create_pdf(df)

        st.download_button(
        label="PDF İndir",
        data=pdf,
        file_name="monthly_prediction.pdf",
        mime="application/pdf"
        )

# -------------------------
# METRICS
# -------------------------

elif menu == "Model Performansı":

    st.metric("ROC-AUC","0.916")
    st.metric("PR-AUC","0.80")
    st.metric("Brier Score","0.105")
    st.metric("Log Loss","0.327")
    st.metric("Forward Accuracy","0.88")

    st.write("Model Version: 1.0")

