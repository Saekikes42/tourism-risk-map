import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import plotly.express as px
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Агент 2.3: Кластеризация Рисков", layout="wide")

DB_PATH = os.getenv("DB_PATH", "postgresql+pg8000://postgres:1234@localhost:5432/tourism_data")

# --- КЭШИРОВАНИЕ ДАННЫХ---
@st.cache(show_spinner=False, allow_output_mutation=True)
def fetch_raw_data():
    try:
        # Пересоздаем engine внутри кэша для безопасности
        temp_engine = create_engine(DB_PATH)
        return pd.read_sql("SELECT * FROM track_points", temp_engine)
    except Exception as e:
        st.error(f"Ошибка загрузки БД: {e}")
        return pd.DataFrame()

def feature_engineering(df):
    """ Создание эвристических признаков риска (Задание 2.3) """
    df_features = df.copy()
    
    df_features['datetime'] = pd.to_datetime(df_features['datetime'], errors='coerce')
    df_features['month'] = df_features['datetime'].dt.month
    
    # 1. Затопление
    is_spring = df_features['month'].isin([3, 4, 5]).astype(int)
    water = df_features['water_counts'].fillna(0)
    max_elev = df_features['elevation'].max() if not df_features['elevation'].empty else 1000
    inv_elev = max_elev - df_features['elevation'].fillna(0)
    df_features['flood_risk'] = is_spring * water * inv_elev
    
    # 2. Пожар
    is_summer = df_features['month'].isin([6, 7, 8]).astype(int)
    temp = df_features['temp'].fillna(0)
    veg = df_features['vegetation_counts'].fillna(0)
    df_features['fire_risk'] = is_summer * temp * veg
    
    # 3. Эвакуация
    dist = df_features['distance'].fillna(0)
    elev = df_features['elevation'].fillna(0)
    df_features['evac_difficulty'] = dist * elev
    
    features_list = ['elevation', 'temp', 'speed', 'distance', 'flood_risk', 'fire_risk', 'evac_difficulty']
    X = df_features[features_list].copy().fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df_features, X, X_scaled

@st.cache(show_spinner=False)
def get_processed_data(df):
    """ Обертка для кэширования признаков """
    return feature_engineering(df)

def evaluate_clusters(X_scaled, labels):
    unique_labels = set(labels)
    if len(unique_labels) < 2 or (len(unique_labels) == 2 and -1 in unique_labels):
        return -1, -1, float('inf')
    sil = silhouette_score(X_scaled, labels)
    cal_har = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    return sil, cal_har, db

st.title("Агент Разметки Опасных Участков (Задачи 2.3 и 2.4)")
st.markdown("Этот интерактивный инструмент использует машинное обучение для автоматизированной разметки зон риска на основе гео-данных маршрутов.")

# 1. Загрузка данных
df_raw = fetch_raw_data()
if df_raw.empty:
    st.warning("База данных пуста или недоступна.")
    st.stop()

st.info(f"Загружено записей: **{len(df_raw)}**")

# 2. Обработка признаков (кэшировано)
df_processed, X_all, _ = get_processed_data(df_raw)

# Инициализация подключения
engine = create_engine(DB_PATH)

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("1. Выбор признаков")
    all_numeric_cols = X_all.columns.tolist()
    selected_features = st.multiselect("Выберите признаки для обучения", 
                                       all_numeric_cols, 
                                       default=['elevation', 'flood_risk', 'fire_risk', 'evac_difficulty'])
    
    st.header("2. Поиск оптимального K")
    plot_optimal = st.button("Показать графики Локтя/Силуэта")
    
    st.header("3. Настройки алгоритмов")
    k_clusters = st.slider("Количество кластеров (K-Means)", 2, 10, 3)
    eps_val = st.slider("Epsilon (DBSCAN)", 0.1, 5.0, 0.5, 0.1)
    min_samples = st.slider("Min Samples (DBSCAN)", 2, 20, 5)
    gmm_components = st.slider("Компоненты (GMM)", 2, 10, 3)
    
    st.markdown("---")
    run_btn = st.button("Запустить Обучение")
    
    if st.button("Очистить результаты"):
        if 'clustering_results' in st.session_state:
            del st.session_state.clustering_results
        st.experimental_rerun()

# --- МЕТОД ЛОКТЯ ---
if plot_optimal:
    st.subheader("Анализ оптимального количества кластеров (K-Means)")
    X_subset = X_all[selected_features].fillna(0)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_subset)
    
    inertias, silhouettes = [], []
    k_range = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_sc)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_sc, km.labels_))
        
    col1, col2 = st.columns(2)
    with col1:
        fig_e, ax = plt.subplots()
        ax.plot(k_range, inertias, 'bo-'); ax.set_title('Метод локтя'); st.pyplot(fig_e)
    with col2:
        fig_s, ax = plt.subplots()
        ax.plot(k_range, silhouettes, 'ro-'); ax.set_title('Анализ силуэта'); st.pyplot(fig_s)

# --- ЛОГИКА ОБУЧЕНИЯ (Session State) ---
if run_btn:
    if not selected_features:
        st.error("Выберите признаки!")
        st.stop()
        
    with st.spinner("Обучение всех моделей..."):
        X_subset = X_all[selected_features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_subset)
        
        # Обучение всех
        km = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto')
        km_labels = km.fit_predict(X_scaled)
        
        db = DBSCAN(eps=eps_val, min_samples=min_samples)
        db_labels = db.fit_predict(X_scaled)
        
        gmm = GaussianMixture(n_components=gmm_components, random_state=42)
        gmm_labels = gmm.fit_predict(X_scaled)
        
        # Оценки
        metrics = {
            "K-Means": evaluate_clusters(X_scaled, km_labels),
            "DBSCAN": evaluate_clusters(X_scaled, db_labels),
            "GMM": evaluate_clusters(X_scaled, gmm_labels)
        }
        
        # PCA для визуализации всех алгоритмов
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        best_name = max(metrics, key=lambda x: metrics[x][0])
        labels_map = {"K-Means": km_labels, "DBSCAN": db_labels, "GMM": gmm_labels}
        
        # Сохранение в состояние
        st.session_state.clustering_results = {
            "metrics": metrics,
            "best_name": best_name,
            "labels_all": labels_map,
            "X_pca": X_pca,
            "selected_features": selected_features,
            "df_res": df_processed.copy()
        }

# --- ВЫВОД РЕЗУЛЬТАТОВ (если есть в состоянии) ---
if 'clustering_results' in st.session_state:
    res = st.session_state.clustering_results
    metrics = res['metrics']
    best_name = res['best_name']
    labels_all = res['labels_all']
    X_pca = res['X_pca']
    
    st.markdown("---")
    st.subheader("Визуализация результатов всех моделей (PCA 2D)")
    
    v_col1, v_col2, v_col3 = st.columns(3)
    for i, (name, labels) in enumerate(labels_all.items()):
        col = [v_col1, v_col2, v_col3][i]
        with col:
            st.write(f"#### {name}")
            fig, ax = plt.subplots(figsize=(6, 5))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=20)
            ax.legend(*scatter.legend_elements(), title="Кластеры")
            ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("---")
    st.subheader("Сравнение метрик")
    
    names = list(metrics.keys())
    sils = [m[0] for m in metrics.values()]
    cals = [m[1] for m in metrics.values()]
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        fig_m1, ax = plt.subplots(); ax.bar(names, sils, color=['#3498db', '#2ecc71', '#e74c3c'])
        ax.set_title('Silhouette Score'); st.pyplot(fig_m1)
    with m_col2:
        fig_m2, ax = plt.subplots(); ax.bar(names, cals, color=['#3498db', '#2ecc71', '#e74c3c'])
        ax.set_title('Calinski-Harabasz Index'); st.pyplot(fig_m2)

    st.success(f"**Вердикт:** Лучшая разметка у **{best_name}**")
    
    # Описание выбранного
    st.subheader(f"Профили кластеров ({best_name})")
    df_res = res['df_res']
    df_res['Cluster'] = res['labels_all'][best_name]
    
    df_stat = df_res[df_res['Cluster'] != -1]
    profiles = df_stat.groupby('Cluster')[res['selected_features']].mean()
    
    p_col1, p_col2, p_spacer = st.columns([2, 1, 2])
    with p_col1:
        st.write("**Средние значения по кластерам:**")
        st.dataframe(profiles.round(2))
    with p_col2:
        st.write("**Размеры кластеров:**")
        size_df = df_res['Cluster'].value_counts().sort_index().reset_index()
        size_df.columns = ['Кластер', 'Кол-во']
        st.dataframe(size_df)
    
    # Диаграмма профилей
    fig_cp, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(res['selected_features']))
    width = 0.8 / len(profiles)
    for i, (idx, row) in enumerate(profiles.iterrows()):
        ax.bar(x + (i - len(profiles)/2)*width + width/2, row.values, width, label=f'C{idx}')
    ax.set_xticks(x); ax.set_xticklabels(res['selected_features'], rotation=45); ax.legend(); st.pyplot(fig_cp)
    
    st.subheader("Карта рисков")
    from streamlit_folium import folium_static
    
    df_map = df_res.dropna(subset=['latitude', 'longitude'])
    if not df_map.empty:
        m = folium.Map(location=[df_map['latitude'].mean(), df_map['longitude'].mean()], zoom_start=7)
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'gray']
        for _, row in df_map.iterrows():
            cid = int(row['Cluster'])
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color=colors[cid % len(colors)] if cid != -1 else 'gray',
                fill=True,
                popup=f"Кластер: {cid}"
            ).add_to(m)
        folium_static(m, width=950, height=500)
    
    if st.button("Сохранить выбранную разметку в БД"):
        with st.spinner("Сохранение..."):
            df_res['risk_cluster'] = df_res['Cluster']
            df_res.to_sql("track_points", engine, if_exists="replace", index=False)
            st.success(f"Разметка от {best_name} сохранена в Postgres!")

