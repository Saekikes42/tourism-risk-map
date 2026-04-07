import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import create_engine
import os

DB_PATH = os.getenv("DB_PATH", "postgresql+pg8000://postgres:1234@localhost:5432/tourism_data")
engine = create_engine(DB_PATH)
OUTPUT_DIR = "analysis_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_agent_1_3():
    print("=== Модуль А. Агенты 1.3 и 1.4: Анализ и предобработка ===")
    
    try:
        df = pd.read_sql("SELECT * FROM track_points", engine)
    except:
        print("Ошибка. Сначала загрузите данные (agent_1_data_loader.py).")
        return
        
    print(f"Загружено {len(df)} записей.")
    
    # 1.3 Определение наиболее значимых признаков (Используем Random Forest & SHAP)
    # Предположим, что мы хотим предсказывать 'step_frequency' (частоту шагов) на основе других факторов
    
    features = ['elevation', 'temp', 'water_counts', 'vegetation_counts', 'distance', 'speed']
    target = 'step_frequency'
    
    # Убедимся, что нет NaN
    df_clean = df[features + [target]].dropna()
    
    X = df_clean[features]
    y = df_clean[target]
    
    # Корреляционная матрица
    plt.figure(figsize=(8, 6))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Корреляционная матрица признаков")
    plt.tight_layout()
    corr_path = os.path.join(OUTPUT_DIR, "correlation_matrix.png")
    plt.savefig(corr_path)
    plt.close()
    
    # Важность признаков через модель
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Feature Importance")
    shap_path = os.path.join(OUTPUT_DIR, "shap_summary.png")
    plt.savefig(shap_path, bbox_inches='tight')
    plt.close()
    
    print(f"Графики анализа сохранены: \n - {corr_path}\n - {shap_path}")
    
    # 1.4 Проверка нормальности распределения (на примере elevation)
    plt.figure(figsize=(10, 5))
    
    # Гистограмма
    plt.subplot(1, 2, 1)
    sns.histplot(df_clean['elevation'], kde=True)
    plt.title("Распределение высоты (Elevation)")
    
    # Q-Q plot (графический метод проверки нормальности)
    from scipy import stats
    plt.subplot(1, 2, 2)
    stats.probplot(df_clean['elevation'], dist="norm", plot=plt)
    plt.title("Q-Q Plot для высоты")
    
    dist_path = os.path.join(OUTPUT_DIR, "elevation_distribution.png")
    plt.tight_layout()
    plt.savefig(dist_path)
    plt.close()
    
    print(f"Отчет по распределению сохранен: {dist_path}")
    print("=== Агент завершил работу ===")

if __name__ == "__main__":
    process_agent_1_3()
