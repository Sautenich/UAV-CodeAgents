import os
import random
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path

# Абсолютный путь к корню проекта
BASE_DIR = Path(__file__).resolve().parent.parent

def find_village_centers(img):
    """Сканирует изображение и ищет зоны с высокой плотностью объектов."""
    w, h = img.size
    potential_zones = []
    step = 50 
    
    for y in range(step, h - step, step):
        for x in range(step, w - step, step):
            crop = img.crop((x-25, y-25, x+25, y+25)).convert("L")
            edges = crop.filter(ImageFilter.FIND_EDGES)
            pixel_data = list(edges.getdata())
            # Ищем зоны, где много краев (постройки)
            if sum(pixel_data) / len(pixel_data) > 12: 
                potential_zones.append((x, y))
    return potential_zones

def add_realistic_fire(base_image, sprite_path):
    img = base_image.copy().convert("RGBA")
    
    # 1. Находим поселения
    zones = find_village_centers(img)
    if not zones:
        return base_image.convert("RGB")
    
    # 2. Выбираем случайное поселение и ищем там самую яркую крышу
    zx, zy = random.choice(zones)
    crop = img.crop((zx-30, zy-30, zx+30, zy+30)).convert("L")
    
    # Размытие убирает одиночные блики
    blurred = crop.filter(ImageFilter.GaussianBlur(radius=3))
    
    # Ищем координаты самой яркой точки
    # getextrema() возвращает (min, max), чтобы найти координаты максимума, используем метод ниже:
    min_val, max_val = blurred.getextrema()
    # Находим координаты всех пикселей с максимальной яркостью
    # В качестве запасного варианта берем центр зоны
    max_loc = (30, 30) 
    # Ищем координаты пикселя с максимальной яркостью
    data = list(blurred.getdata())
    max_idx = data.index(max_val)
    max_loc = (max_idx % 60, max_idx // 60)
    
    x = zx - 30 + max_loc[0]
    y = zy - 30 + max_loc[1]
    
    # Защита от слишком ярких объектов (бликов)
    r, g, b = img.convert("RGB").getpixel((x, y))
    if (r + g + b) > 650:
        x, y = zx, zy
            
    # 3. Наложение огня
    fire = Image.open(sprite_path).convert("RGBA").resize((25, 25))
    img.paste(fire, (x - 12, y - 12), fire)
    
    # 4. Добавление дыма
    smoke = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(smoke)
    d.ellipse([x - 20, y - 30, x + 20, y + 10], fill=(40, 40, 40, 180))
    smoke = smoke.filter(ImageFilter.GaussianBlur(radius=10))
    
    combined = Image.alpha_composite(img, smoke)
    return combined.convert("RGB")

def process_folder():
    input_folder = BASE_DIR / "data" / "mission_images"
    output_folder = BASE_DIR / "data" / "mission_images_fire"
    sprite_path = BASE_DIR / "src" / "assets" / "fire_sprite.png"
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Обработка: {filename}...")
            img = Image.open(input_folder / filename).convert("RGBA")
            fire_img = add_realistic_fire(img, sprite_path)
            fire_img.save(output_folder / f"fire_{filename}", "JPEG", quality=90)
            
    print("Готово!")

if __name__ == "__main__":
    process_folder()