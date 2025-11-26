import os
import time
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

# Gemini
import google.generativeai as genai

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="AQI Chatbot & Dashboard", layout="wide")
st.title("Air Pollution Index Chatbot & Dashboard")

# Sidebar
st.sidebar.header("Settings")
city = st.sidebar.text_input("City (OpenAQ compatible)", value="Jaipur")
country = st.sidebar.text_input("Country code (optional)", value="")
hours = st.sidebar.slider("Hours lookback", min_value=6, max_value=72, value=24, step=6)

# Gemini setup
GEMINI_KEY = os.getenv("Your api key here")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
else:
    st.warning("Set GEMINI_API_KEY env var")

model = None
try:
    if GEMINI_KEY:
        model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    st.error(f"Gemini init error: {e}")

# ---------------------------
# Helper functions
# ---------------------------
def fetch_openaq(city, country, hours):
    """
    Fetch recent measurements from OpenAQ v2 API.
    We aggregate PM2.5/PM10, NO2, O3, SO2 if available.
    """
    base = "https://api.openaq.org/v2/measurements"
    params = {
        "city": city,
        "limit": 1000,
        "page": 1,
        "offset": 0,
        "sort": "desc",
        "order_by": "datetime",
        "date_from": pd.Timestamp.utcnow() - pd.Timedelta(hours=hours),
        "date_to": pd.Timestamp.utcnow(),
    }
    if country.strip():
        params["country"] = country.strip()
    # Convert timestamps to ISO
    params["date_from"] = params["date_from"].isoformat() + "Z"
    params["date_to"] = params["date_to"].isoformat() + "Z"

    r = requests.get(base, params=params, timeout=20)
    r.raise_for_status()
    data = r.json().get("results", [])
    if not data:
        return pd.DataFrame()
    rows = []
    for item in data:
        rows.append({
            "datetime": item.get("date", {}).get("utc"),
            "location": item.get("location"),
            "parameter": item.get("parameter"),
            "value": item.get("value"),
            "unit": item.get("unit"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def indicative_aqi(pm25):
    """
    Very simplified indicative AQI banding (not official).
    Returns category string based on PM2.5 (µg/m3).
    """
    if pd.isna(pm25):
        return "Unknown"
    if pm25 <= 12:
        return "Good"
    elif pm25 <= 35.4:
        return "Moderate"
    elif pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups"
    elif pm25 <= 150.4:
        return "Unhealthy"
    elif pm25 <= 250.4:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def compose_context_summary(df):
    """
    Create a textual summary of recent pollution for chat context.
    """
    if df.empty:
        return "No recent measurements found."
    pivot = (
        df.pivot_table(index="datetime", columns="parameter", values="value", aggfunc="mean")
        .sort_index()
    )
    latest = pivot.tail(1)
    latest_vals = latest.to_dict(orient="records")[0] if not latest.empty else {}
    pm25 = latest_vals.get("pm25")
    aqi_band = indicative_aqi(pm25)
    parts = [f"Latest indicative AQI (by PM2.5): {aqi_band}"]
    for k in ["pm25", "pm10", "no2", "o3", "so2"]:
        if k in latest_vals and pd.notna(latest_vals[k]):
            parts.append(f"{k.upper()}={latest_vals[k]:.1f}")
    return "; ".join(parts)

# ---------------------------
# Layout: Dashboard
# ---------------------------
with st.container():
    st.subheader("City air quality overview")
    col1, col2 = st.columns([3, 2])

    with col1:
        status = st.empty()
        try:
            df = fetch_openaq(city, country, hours)
            if df.empty:
                status.warning("No data for selected inputs.")
            else:
                status.success(f"Fetched {len(df)} measurements.")
                # Trend chart: PM2.5/PM10
                pm_df = df[df["parameter"].isin(["pm25", "pm10"])]
                if not pm_df.empty:
                    fig = px.line(
                        pm_df,
                        x="datetime",
                        y="value",
                        color="parameter",
                        title="PM2.5 and PM10 over time",
                        markers=True,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                # Pollutant distribution
                agg = df.groupby("parameter")["value"].mean().reset_index()
                fig2 = px.bar(
                    agg, x="parameter", y="value", title="Average pollutant concentrations"
                )
                st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            status.error(f"Data fetch error: {e}")

    with col2:
        st.subheader("Latest indicative AQI")
        ctx_summary = compose_context_summary(df if 'df' in locals() else pd.DataFrame())
        st.metric(label="Summary", value=ctx_summary.split(';')[0] if ctx_summary else "Unknown")
        st.caption("Indicative band based on latest PM2.5; not an official AQI.")

# ---------------------------
# Layout: Chatbot
# ---------------------------
st.subheader("Ask the AQI chatbot")
st.caption("Get explanations about AQI, health guidance, and pollutant differences.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_msg = st.text_input("Your question (e.g., 'Is it safe to run outside today?')", value="")
send = st.button("Send")

def system_prompt():
    return (
        "You are an air quality assistant. Explain AQI categories, health effects, and safe actions "
        "in clear, concise terms. Use the provided context when relevant. If the AQI is 'Unknown', "
        "suggest checking more locations or time ranges. Avoid medical advice; keep it general."
    )

if send and user_msg.strip():
    st.session_state.chat_history.append(("user", user_msg))
    # Compose context to ground responses
    context = ctx_summary
    final_prompt = (
        f"{system_prompt()}\n\n"
        f"Context for city '{city}': {context}\n\n"
        f"User: {user_msg}\nAssistant:"
    )
    try:
        if model:
            resp = model.generate_content(final_prompt)
            answer = resp.text.strip() if hasattr(resp, "text") else "No response."
        else:
            answer = "Gemini is not configured. Please set GEMINI_API_KEY."
    except Exception as e:
        answer = f"Model error: {e}"
    st.session_state.chat_history.append(("assistant", answer))

# Show chat
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Assistant:** {msg}")

# Footer
st.markdown("---")
st.caption("Data source: OpenAQ. AQI band is indicative and simplified. Gemini powers the chatbot.")
