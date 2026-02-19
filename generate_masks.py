import cv2
import numpy as np
import os
import random

NUM_MASKS = 1000  # сколько масок сгенерировать
HEIGHT, WIDTH = 256, 256  # размер маски (как в твоём датасете)
MIN_COVERAGE = 0.05  # минимальное покрытие
MAX_COVERAGE = 0.30  # максимальное
SAVE_DIR = "../data/masks/custom_masks"
os.makedirs(SAVE_DIR, exist_ok=True)

for i in range(NUM_MASKS):
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    
    # Сгенерируем случайные линии (царапины)
    num_lines = random.randint(3, 15)
    for _ in range(num_lines):
        x1, y1 = random.randint(0, WIDTH), random.randint(0, HEIGHT)
        x2, y2 = random.randint(0, WIDTH), random.randint(0, HEIGHT)
        thickness = random.randint(3, 10)
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
    
    # Добавим блобы (круглые/овальные дыры)
    num_blobs = random.randint(1, 5)
    for _ in range(num_blobs):
        center = (random.randint(0, WIDTH), random.randint(0, HEIGHT))
        axes = (random.randint(10, 50), random.randint(10, 50))
        angle = random.randint(0, 360)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)  # -1 = заполненный
    
    # Проверим покрытие
    coverage = np.sum(mask == 255) / (HEIGHT * WIDTH)
    if coverage < MIN_COVERAGE or coverage > MAX_COVERAGE:
        continue  # пропустим, если не в диапазоне
    
    cv2.imwrite(os.path.join(SAVE_DIR, f"mask_{i:04d}.png"), mask)
    print(f"Сгенерирована маска {i+1}/{NUM_MASKS}, покрытие {coverage:.2%}")

print("Готово! Маски в ../data/masks/custom_masks")