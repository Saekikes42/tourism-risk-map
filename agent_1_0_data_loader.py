import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import contextily as cx
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime
import os
import hashlib

# Конфигурация
DB_PATH = os.getenv("DB_PATH", "postgresql+pg8000://postgres:1234@localhost:5432/tourism_data")
CSV_PATH = "C:\\Users\\2.0\\.gemini\\antigravity\\scratch\\map\\df_with_regions.csv"
IMAGES_DIR = "track_images"

# Создаем папку для картинок, если её нет
os.makedirs(IMAGES_DIR, exist_ok=True)

# Инициализация подключения к БД
engine = create_engine(DB_PATH)

def load_data_from_csv(csv_path):
    print("Загрузка данных из CSV...")
    df = pd.read_csv(csv_path)
    # Удаляем пустой первый столбец-индекс, если он есть
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df

def generate_track_map_image(df_track, track_id):
    """
    Генерирует изображение карты с наложенным треком.
    """
    image_path = os.path.join(IMAGES_DIR, f"track_{track_id}.png")
    
    # Если картинка уже есть, не перерисовываем (экономим время)
    if os.path.exists(image_path):
        return image_path
        
    print(f"Генерация карты для трека {track_id}...")
    
    # Создаем геометрию из точек
    geometry = [Point(xy) for xy in zip(df_track.longitude, df_track.latitude)]
    gdf_points = gpd.GeoDataFrame(df_track, geometry=geometry, crs="EPSG:4326")
    
    # Конвертируем в проекцию Web Mercator (EPSG:3857) для contextily
    gdf_points = gdf_points.to_crs(epsg=3857)
    
    # Создаем линию трека
    if len(gdf_points) > 1:
        line = LineString(gdf_points.geometry.tolist())
        gdf_line = gpd.GeoDataFrame(geometry=[line], crs="EPSG:3857")
    else:
        gdf_line = gdf_points

    # Рисуем карту
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_line.plot(ax=ax, linewidth=3, edgecolor='red')
    
    # Добавляем векторную подложку (OpenStreetMap)
    try:
        cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
    except Exception as e:
        print(f"Не удалось загрузить подложку для трека {track_id}: {e}")
        
    ax.set_axis_off()
    plt.title(f"Маршрут трека: {track_id}")
    plt.tight_layout()
    plt.savefig(image_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    return image_path

def save_to_database(df, engine):
    """
    Сохраняет данные в базу. Реализована логика избежания дублирования.
    """
    print("Подготовка к сохранению в базу данных...")
    
    # 1. Сохранение точек
    # Для идемпотентности, перезаписываем таблицу points (или можно использовать UPSERT)
    # В рамках задания проще обновлять/добавлять только новые.
    # Так как у нас теперь PostgreSQL, используем метод pandas to_sql с if_exists='replace'
    # В реальном проекте лучше 'append' и проверять уникальность.
    df.to_sql("track_points", engine, if_exists="replace", index=False)
    print("Точки трека успешно сохранены.")

def process_agent_1_1():
    print("=== Модуль А. Агент 1.1: Загрузка и генерация карт ===")
    df = load_data_from_csv(CSV_PATH)
    
    # Добавим колонку для пути к изображению
    df['map_image_path'] = ""
    
    track_ids = df['track_id'].unique()
    print(f"Найдено треков: {len(track_ids)}")
    
    for tid in track_ids:
        df_track = df[df['track_id'] == tid]
        img_path = generate_track_map_image(df_track, tid)
        # Обновляем инфу в основном датафрейме
        df.loc[df['track_id'] == tid, 'map_image_path'] = img_path
        
    save_to_database(df, engine)
    print("=== Задание 1.1 завершено! ===")

if __name__ == "__main__":
    process_agent_1_1()
