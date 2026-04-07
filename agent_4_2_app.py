import streamlit as st
import requests
import pydeck as pdk
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict_risk"


st.title("Оценка маршрутов (Модуль Г)")
st.markdown("Модуль 4.2: Введите планируемые параметры для запроса API (4.1)")

with st.sidebar:
    st.header("Опции маршрута")
    month = st.slider("Месяц похода", 1, 12, 6)
    temp = st.slider("Ожидаемая темп. (°C)", -30.0, 40.0, 15.0)
    
    elevation = st.number_input("Текущая высота (м)", 0.0, 8000.0, 500.0)
    distance = st.number_input("Пройденная дистанция (км)", 0.0, 100.0, 5.0)
    speed = st.slider("Скорость (м/с)", 0.5, 5.0, 1.5)
    
    water = st.slider("Водоемов (в радиусе 500м)", 0, 10, 1)
    veg = st.slider("Растительности (лес)", 0, 10, 5)
    
    st.markdown("---")
    lat = st.number_input("Широта (Lat)", 50.0, 60.0, 55.75)
    lon = st.number_input("Долгота (Lon)", 30.0, 50.0, 37.61)

if st.button("ИИ: Предсказать уровень риска"):
    payload = {
        "elevation": elevation, "temp": temp, "speed": speed,
        "distance": distance, "water_counts": water, 
        "vegetation_counts": veg, "month": month
    }
    
    try:
        with st.spinner("Запрос к API FastAPI..."):
            response = requests.post(API_URL, json=payload, timeout=5)
            
        if response.status_code == 200:
            data = response.json()
            risk = data["risk_cluster"]
            conf = data["confidence"]
            
            colors = {
                0: [0, 255, 0, 160],   # Низкий
                1: [255, 165, 0, 160], # Средний
                2: [255, 0, 0, 160]    # Высокий
            }
            labels = {
                0: "Безопасно (Зеленая зона)",
                1: "Средний: Риск паводка или трудной эвакуации",
                2: "Высокий риск: Осторожно (Пожароопасно / Сложный рельеф)"
            }
            
            st.success(f"Анализ API завершен! Точность уверенности модели: {conf*100:.1f}%")
            st.subheader(f"Уровень: {labels.get(risk, 'Неизвестно')}")
            
            map_data = pd.DataFrame([{"lat": lat, "lon": lon, "risk": str(risk)}])
            
            # Используем Plotly для надежности 
            import plotly.express as px
            color_map = {"0": "green", "1": "orange", "2": "red"}
            
            map_fig = px.scatter_mapbox(
                map_data, lat="lat", lon="lon", color="risk",
                color_discrete_map=color_map, zoom=10, height=400
            )
            # Чтобы точки были крупнее, укажем фиксированный размер
            map_fig.update_traces(marker=dict(size=20))
            map_fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
            
            st.plotly_chart(map_fig, use_container_width=True)
        else:
            st.error(f"Ошибка API: {response.text}")
    except Exception as e:
        st.error(f"API недоступно. Убедитесь, что сервер uvicorn (agent_4_1_api.py) запущен! Ошибка: {e}")
