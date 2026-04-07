import pandas as pd
import overpy
from sqlalchemy import create_engine
import time
import os

DB_PATH = os.getenv("DB_PATH", "postgresql+pg8000://postgres:1234@localhost:5432/tourism_data")
engine = create_engine(DB_PATH)
api = overpy.Overpass()

def get_surroundings_overpass(lat, lon, radius=500):
    """
    Получает информацию об окружении в радиусе `radius` метров.
    Возвращает строку с типом местности или пустую строку.
    """
    curr_time = time.time()
    # Формируем запрос (ищем леса, воду, жилые зоны)
    query = f"""
    [out:json];
    (
      way["landuse"="forest"](around:{radius},{lat},{lon});
      way["natural"="water"](around:{radius},{lat},{lon});
      way["natural"="wood"](around:{radius},{lat},{lon});
      way["landuse"="residential"](around:{radius},{lat},{lon});
      node["place"="village"](around:{radius},{lat},{lon});
      node["natural"="spring"](around:{radius},{lat},{lon});
    );
    out body;
    """
    
    features = []
    try:
        result = api.query(query)
        for way in result.ways:
            if "landuse" in way.tags and way.tags["landuse"] == "forest":
                features.append("лес")
            if "natural" in way.tags and way.tags["natural"] == "wood":
                features.append("лес")
            if "natural" in way.tags and way.tags["natural"] == "water":
                features.append("вода")
            if "landuse" in way.tags and way.tags["landuse"] == "residential":
                features.append("населенный пункт")
                
        for node in result.nodes:
            if "place" in node.tags and node.tags["place"] == "village":
                features.append("поселок")
            if "natural" in node.tags and node.tags["natural"] == "spring":
                features.append("источник воды")
                
    except overpy.exception.OverpassTooManyRequests:
        print("Слишком много запросов к Overpass. Пауза...")
        time.sleep(5)
    except Exception as e:
        print(f"Ошибка Overpass: {e}")
        
    # Убираем дубликаты
    features = list(set(features))
    return ", ".join(features) if features else "открытая местность"

def process_agent_1_2():
    print("=== Модуль А. Агент 1.2: Сбор окружения (500м) ===")
    
    # Читаем данные, которые сохранил Агент 1.1
    try:
        df = pd.read_sql("SELECT * FROM track_points", engine)
    except Exception as e:
        print("Ошибка чтения БД. Сначала запустите agent_1_data_loader.py")
        return
        
    print(f"Загружено {len(df)} точек. Начинаем обогащение данных (это может занять время)...")
    
    # Добавляем колонку, если её нет
    if 'environment_type' not in df.columns:
        df['environment_type'] = ""
        
    # Чтобы не делать тысячи запросов к Overpass (ведь точки стоят через метр),
    # будем брать каждую N-ную точку трека и применять для отрезка.
    # Но для конкурсного задания сделаем выборку только уникальных геопозиций с округлением.
    
    # Округлим координаты до 3 знаков (это примерно 110 метров разницы)
    df['lat_round'] = df['latitude'].round(3)
    df['lon_round'] = df['longitude'].round(3)
    
    cache = {}
    
    for idx, row in df.iterrows():
        # Если тип местности уже заполнен, пропускаем
        if pd.notna(row['environment_type']) and row['environment_type'] != "":
            continue
            
        coord_key = (row['lat_round'], row['lon_round'])
        
        if coord_key in cache:
            env_val = cache[coord_key]
        else:
            env_val = get_surroundings_overpass(row['latitude'], row['longitude'], radius=500)
            cache[coord_key] = env_val
            time.sleep(1) # Защита от бана в бесплатном Overpass API
            if len(cache) % 10 == 0:
                print(f"Обработано локаций: {len(cache)}")
                
        df.at[idx, 'environment_type'] = env_val
        
    # Чистим вспомогательные столбцы
    df = df.drop(columns=['lat_round', 'lon_round'])
    
    # Перезаписываем в базу
    df.to_sql("track_points", engine, if_exists="replace", index=False)
    print("=== Задание 1.2 завершено! Окружение успешно определено. ===")

if __name__ == "__main__":
    process_agent_1_2()
