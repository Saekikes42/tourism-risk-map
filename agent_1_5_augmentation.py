import albumentations as A
import cv2
import os
import glob

IMAGES_DIR = "track_images"
AUG_DIR = "track_images_augmented"
os.makedirs(AUG_DIR, exist_ok=True)

def augment_images():
    print("=== Модуль А. Агент 1.5: Расширение данных (Аугментация) ===")
    
    # Определяем цепочку трансформаций
    transform = A.Compose([
        A.HorizontalFlip(p=0.5), # Зеркальное отражение
        A.RandomBrightnessContrast(p=0.8), # Изменение яркости и контраста
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=45, p=0.7), # Сдвиг, масштаб, поворот
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5), # Изменение цвета
    ])

    image_paths = glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    if not image_paths:
        print("Картинки не найдены. Выполните agent_1_data_loader.py")
        return
        
    print(f"Найдено {len(image_paths)} изображений. Начинаем аугментацию...")
    
    for count, path in enumerate(image_paths):
        image = cv2.imread(path)
        if image is None:
            continue
            
        # Зациклим создание 3 аугментаций на каждое изображение
        for i in range(3):
            # OpenCV читает в формате BGR, переводим в RGB для Albumentations (если важно для цвета)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented = transform(image=image_rgb)
            aug_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
            
            # Сохранение файла
            base_name = os.path.basename(path).replace(".png", "")
            out_path = os.path.join(AUG_DIR, f"{base_name}_aug_{i}.png")
            cv2.imwrite(out_path, aug_img)
            
    print(f"Аугментация завершена! Результаты помещены в папку '{AUG_DIR}'")

if __name__ == "__main__":
    augment_images()
