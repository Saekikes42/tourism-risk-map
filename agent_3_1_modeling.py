import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score
from sklearn.preprocessing import label_binarize

st.set_page_config(page_title="Агент 3.1: Классификация Рисков", layout="wide")

DB_PATH = os.getenv("DB_PATH", "postgresql+pg8000://postgres:1234@localhost:5432/tourism_data")
engine = create_engine(DB_PATH)
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Кэширование данных
@st.cache(show_spinner=False, allow_output_mutation=True)
def load_data():
    try:
        df = pd.read_sql("SELECT * FROM track_points", engine)
        if 'risk_cluster' not in df.columns:
            return None
        return df
    except Exception as e:
        st.error(f"Ошибка БД: {e}")
        return None

def preprocess_data(df):
    df_clean = df.dropna(subset=['risk_cluster']).copy()
    df_clean['datetime'] = pd.to_datetime(df_clean['datetime'], errors='coerce')
    df_clean['month'] = df_clean['datetime'].dt.month.fillna(6)
    
    features = ['elevation', 'temp', 'speed', 'distance', 'water_counts', 'vegetation_counts', 'month']
    X = df_clean[features].fillna(0)
    y = df_clean['risk_cluster'].astype(int)
    return X, y, features

st.title("Агент 3.1: Машинное обучение (Классификация)")
st.markdown("Этот модуль обучает модели классификации для предсказания уровня риска на основе исторических данных.")

df = load_data()
if df is None:
    st.warning("Данные не найдены или столбец risk_cluster отсутствует. Запустите сначала Модуль 2.3.")
    st.stop()

st.info(f"Загружено записей для обучения: {len(df)}")

X, y, feature_list = preprocess_data(df)
classes = np.unique(y)
is_binary = len(classes) == 2

with st.sidebar:
    st.header("Настройки моделей")
    
    st.subheader("Параметры KNN")
    knn_neighbors = st.slider("Количество соседей (K)", 1, 20, 5)
    
    st.subheader("Параметры SVC")
    svc_kernel = st.selectbox("Ядро (Kernel)", ["linear", "poly", "rbf", "sigmoid"], index=2)
    svc_c = st.number_input("Регуляризация (C)", 0.01, 10.0, 1.0)
    
    run_train = st.button("Запустить обучение всех моделей")

if run_train:
    with st.spinner("Обучение моделей классификации"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Инициализация моделей
        models = {
            "KNN": KNeighborsClassifier(n_neighbors=knn_neighbors),
            "SVC": SVC(kernel=svc_kernel, C=svc_c, probability=True, random_state=42),
            "GaussianNB": GaussianNB()
        }
        
        results = {}
        for name, clf in models.items():
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            probs = clf.predict_proba(X_test_scaled) if hasattr(clf, "predict_proba") else None
            results[name] = {
                "acc": accuracy_score(y_test, y_pred),
                "model": clf,
                "y_pred": y_pred,
                "y_proba": probs
            }
            
        # Сохранение в состояние
        st.session_state.model_training_results = {
            "results": results,
            "X_test_scaled": X_test_scaled,
            "y_test": y_test,
            "classes": classes,
            "is_binary": is_binary,
            "X_all": X,
            "y_all": y
        }

# --- ВЫВОД РЕЗУЛЬТАТОВ (из состояния) ---
if 'model_training_results' in st.session_state:
    tr = st.session_state.model_training_results
    results = tr["results"]
    X_test_scaled = tr["X_test_scaled"]
    y_test = tr["y_test"]
    classes = tr["classes"]
    is_binary = tr["is_binary"]
    
    st.markdown("---")
    st.header("Результаты обучения")
    
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            st.subheader(f"Модель: {name}")
            st.metric("Accuracy", f"{res['acc']:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, res["y_pred"])
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax_cm)
            ax_cm.set_title(f'Confusion Matrix: {name}')
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('True')
            st.pyplot(fig_cm)
            plt.close(fig_cm)

    # ROC-кривые
    st.markdown("---")
    st.header("Анализ ROC-кривых")
    
    r_col1, r_col2, r_col3 = st.columns(3)
    r_cols = [r_col1, r_col2, r_col3]
    fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
    lw = 2
    
    for i, (name, res) in enumerate(results.items()):
        clf = res["model"]
        with r_cols[i]:
            st.write(f"#### ROC: {name}")
            if res["y_proba"] is not None:
                if is_binary:
                    fpr, tpr, _ = roc_curve(y_test, res["y_proba"][:, 1], pos_label=classes[1])
                else:
                    idx_1 = np.where(classes == 1)[0]
                    target_idx = idx_1[0] if len(idx_1) > 0 else 0
                    y_test_bin = label_binarize(y_test, classes=classes)[:, target_idx]
                    fpr, tpr, _ = roc_curve(y_test_bin, res["y_proba"][:, target_idx])
                
                roc_auc = auc(fpr, tpr)
                fig_ind, ax_ind = plt.subplots(figsize=(5, 4))
                ax_ind.plot(fpr, tpr, color='darkorange', lw=lw, label=f'AUC = {roc_auc:.2f}')
                ax_ind.plot([0, 1], [0, 1], color='navy', linestyle='--')
                ax_ind.legend(loc="lower right")
                st.pyplot(fig_ind)
                plt.close(fig_ind)
                ax_comp.plot(fpr, tpr, lw=lw, label=f'{name} (AUC = {roc_auc:.2f})')
            else:
                st.warning("Вероятности не поддерживаются")

    st.markdown("---")
    st.subheader("Сравнение ROC-кривых всех моделей")
    ax_comp.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax_comp.legend()
    st.pyplot(fig_comp)
    fig_comp.savefig("ROC.png")
    plt.close(fig_comp)
    
    # Сравнение точности
    st.markdown("---")
    metrics_k = list(results.keys())
    scores_k = [res["acc"] for res in results.values()]
    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
    ax_bar.bar(metrics_k, scores_k, color=['blue', 'green', 'orange'])
    for i, v in enumerate(scores_k):
        ax_bar.text(i, v + 0.01, f'{v:.4f}', ha='center')
    st.pyplot(fig_bar)
    plt.close(fig_bar)
    
    # Сохранение лучшей модели
    best_name = max(results, key=lambda x: results[x]["acc"])
    st.success(f"Наилучшая модель: {best_name}")
    
    if st.button(f"Зафиксировать и сохранить модель {best_name}"):
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', results[best_name]["model"])
        ])
        # Обучаем на всем наборе для продакшена
        pipe.fit(tr["X_all"], tr["y_all"])
        joblib.dump(pipe, os.path.join(MODELS_DIR, "risk_classifier.pkl"))
        st.success("Файловая запись успешно обновлена: models/risk_classifier.pkl")

st.markdown("---")
st.subheader("Интеграция с Airflow")
st.write("Для реализации непрерывного обучения (continuous training) с помощью Airflow рекомендуется следующий подход:")
st.write("1. Создать DAG, который запускается по расписанию (например, ежедневно).")
st.write("2. Первый шаг DAG: проверка наличия новых данных в базе.")
st.write("3. Второй шаг DAG: запуск Python-скрипта (через PythonOperator), который выполняет те же шаги, что и это приложение (обучение, выбор лучшей модели).")
st.write("4. Третий шаг DAG: автоматическая замена файла .pkl, если метрики новой модели выше текущей продовой версии.")
