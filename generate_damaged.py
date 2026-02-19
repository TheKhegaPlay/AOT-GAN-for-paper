# import os
# import cv2
# import numpy as np
# import random
# from pathlib import Path

# # Пути (относительно корня репозитория)
# ORIGINAL_DIR = Path("../data/val/val_256")
# MASK_DIR = Path("../data/masks/test_mask/mask/testing_mask_dataset")  # папка с распакованными масками NVIDIA
# DAMAGED_DIR = Path("../data/images/val_damaged")
# USED_MASK_DIR = Path("../data/masks/val_used")

# DAMAGED_DIR.mkdir(exist_ok=True, parents=True)
# USED_MASK_DIR.mkdir(exist_ok=True, parents=True)

# # Соберём все маски рекурсивно
# all_masks = list(MASK_DIR.rglob("*.png")) + list(MASK_DIR.rglob("*.jpg"))
# print(f"Найдено масок: {len(all_masks)}")

# # Берём все оригиналы (или ограничим количество)
# original_images = list(ORIGINAL_DIR.rglob("*.jpg"))
# random.shuffle(original_images)
# original_images = original_images[:500]  # например 500 штук

# def get_mask_coverage(mask_path):
#     mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
#     if mask is None:
#         return 0
#     # Бинаризуем и считаем долю белого (255)
#     mask = (mask > 127).astype(np.uint8)
#     coverage = np.sum(mask) / (mask.shape[0] * mask.shape[1])
#     return coverage

# # Пример: фильтруем только маски 0.05–0.30
# filtered_masks = []
# for m in all_masks:
#     cov = get_mask_coverage(m)
#     if 0.05 <= cov <= 0.30:
#         filtered_masks.append(m)

# print(f"Отфильтровано масок 5–30%: {len(filtered_masks)}")

# for idx, orig_path in enumerate(original_images):
#     img = cv2.imread(str(orig_path))
#     if img is None:
#         continue

#     # Случайная маска
#     mask_path = random.choice(filtered_masks)
#     mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
#     if mask is None:
#         continue

#     # Приведём маску к размеру изображения
#     mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

#     # Бинаризуем (если не 0/255)
#     _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

#     # Повреждённое изображение: где маска 255 — чёрный
#     damaged = img.copy()
#     damaged[mask == 255] = 0

#     # Сохраняем
#     base_name = orig_path.stem
#     cv2.imwrite(str(DAMAGED_DIR / f"{base_name}_damaged.jpg"), damaged)
#     cv2.imwrite(str(USED_MASK_DIR / f"{base_name}_mask.png"), mask)
#     cv2.imwrite(str(DAMAGED_DIR / f"{base_name}_orig.jpg"), img)  # для сравнения

#     print(f"Обработано {idx+1}/{len(original_images)}: {base_name}")

# print("Готово! Повреждённые изображения в data/images/val_damaged")


















# import os
# import cv2
# import numpy as np
# import random
# import shutil
# from pathlib import Path

# # Пути (относительно корня репозитория)
# ORIGINAL_DIR = Path("../data/val/val_256")
# MASK_DIR = Path("../data/masks/test_mask/mask/testing_mask_dataset")
# DAMAGED_DIR = Path("../data/images/val_damaged")
# USED_MASK_DIR = Path("../data/masks/val_used")

# # Очищаем старые файлы перед новой генерацией
# if DAMAGED_DIR.exists():
#     shutil.rmtree(DAMAGED_DIR)
# if USED_MASK_DIR.exists():
#     shutil.rmtree(USED_MASK_DIR)

# DAMAGED_DIR.mkdir(exist_ok=True, parents=True)
# USED_MASK_DIR.mkdir(exist_ok=True, parents=True)

# # Соберём все маски рекурсивно
# all_masks = list(MASK_DIR.rglob("*.png")) + list(MASK_DIR.rglob("*.jpg"))
# print(f"Найдено масок всего: {len(all_masks)}")

# if len(all_masks) == 0:
#     print("ОШИБКА: маски не найдены! Проверь путь MASK_DIR")
#     exit()

# # Фильтрация масок по покрытию 5–20% (можно изменить)
# filtered_masks = []
# coverages = []
# for m in all_masks:
#     mask_raw = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
#     if mask_raw is None:
#         continue
#     mask_bin = (mask_raw > 10).astype(np.uint8)
#     coverage = np.sum(mask_bin) / (mask_raw.shape[0] * mask_raw.shape[1])
#     if 0.05 <= coverage <= 0.20:
#         filtered_masks.append(m)
#         coverages.append(coverage)

# print(f"Отфильтровано масок 5–20%: {len(filtered_masks)}")
# if coverages:
#     print(f"Среднее покрытие отфильтрованных масок: {np.mean(coverages):.2%}")
# else:
#     print("ВНИМАНИЕ: после фильтрации масок не осталось! Расширь диапазон до 0.01–0.40")
#     exit()

# # Оригиналы
# original_images = list(ORIGINAL_DIR.rglob("*.jpg"))
# random.shuffle(original_images)
# original_images = original_images[:200]  # для теста — потом увеличь

# print(f"Обрабатываем {len(original_images)} изображений")

# processed_count = 0
# for idx, orig_path in enumerate(original_images):
#     img = cv2.imread(str(orig_path))
#     if img is None:
#         print(f"Не удалось прочитать изображение: {orig_path}")
#         continue

#     # Случайная маска из отфильтрованных
#     mask_path = random.choice(filtered_masks)
#     mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
#     if mask is None:
#         print(f"Не удалось прочитать маску: {mask_path}")
#         continue

#     # Ресайз маски
#     mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

#     # Бинаризация
#     _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

#     # Повреждённое изображение: дыры белого цвета (как ожидает модель)
#     damaged = img.copy()
#     damaged[mask == 255] = 255

#     # Сохраняем
#     base_name = orig_path.stem
#     cv2.imwrite(str(DAMAGED_DIR / f"{base_name}_damaged.jpg"), damaged)
#     cv2.imwrite(str(USED_MASK_DIR / f"{base_name}_mask.png"), mask)
#     cv2.imwrite(str(DAMAGED_DIR / f"{base_name}_orig.jpg"), img)  # для сравнения

#     processed_count += 1
#     print(f"Обработано {processed_count}/{len(original_images)}: {base_name}")

# print(f"Готово! Обработано изображений: {processed_count}")
# print(f"Повреждённые изображения в: {DAMAGED_DIR}")
# print(f"Маски сохранены в: {USED_MASK_DIR}")

























import os
import cv2
import numpy as np
import random
import shutil
from pathlib import Path

# Пути (относительно корня репозитория)
ORIGINAL_DIR = Path("../data/train/data_256")
MASK_DIR = Path("../data/masks/test_mask/mask/testing_mask_dataset")
DAMAGED_DIR = Path("../data/images/train_damaged")
USED_MASK_DIR = Path("../data/masks/train_used")

# Очищаем старые файлы перед новой генерацией
if DAMAGED_DIR.exists():
    shutil.rmtree(DAMAGED_DIR)
if USED_MASK_DIR.exists():
    shutil.rmtree(USED_MASK_DIR)

DAMAGED_DIR.mkdir(exist_ok=True, parents=True)
USED_MASK_DIR.mkdir(exist_ok=True, parents=True)

# Соберём все маски рекурсивно
all_masks = list(MASK_DIR.rglob("*.png")) + list(MASK_DIR.rglob("*.jpg"))
print(f"Найдено масок всего: {len(all_masks)}")

if len(all_masks) == 0:
    print("ОШИБКА: маски не найдены! Проверь путь MASK_DIR")
    exit()

# Фильтрация масок по покрытию 5–20% (можно изменить)
filtered_masks = []
coverages = []
for m in all_masks:
    mask_raw = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        continue
    mask_bin = (mask_raw > 10).astype(np.uint8)
    coverage = np.sum(mask_bin) / (mask_raw.shape[0] * mask_raw.shape[1])
    if 0.05 <= coverage <= 0.20:
        filtered_masks.append(m)
        coverages.append(coverage)

print(f"Отфильтровано масок 5–20%: {len(filtered_masks)}")
if coverages:
    print(f"Среднее покрытие отфильтрованных масок: {np.mean(coverages):.2%}")
else:
    print("ВНИМАНИЕ: после фильтрации масок не осталось! Расширь диапазон до 0.01–0.40")
    exit()

# Оригиналы
original_images = list(ORIGINAL_DIR.rglob("*.jpg"))
random.shuffle(original_images)
original_images = original_images[:500]  # для теста — потом увеличь

print(f"Обрабатываем {len(original_images)} изображений")

processed_count = 0
for idx, orig_path in enumerate(original_images):
    img = cv2.imread(str(orig_path))
    if img is None:
        continue

    mask_path = random.choice(filtered_masks)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue

    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    damaged = img.copy()
    damaged[mask == 255] = 255  # белые дыры

    base_name = orig_path.stem
    cv2.imwrite(str(DAMAGED_DIR / f"{base_name}.jpg"), damaged)
    cv2.imwrite(str(USED_MASK_DIR / f"{base_name}.png"), mask)

    print(f"Обработано {idx+1}/{len(original_images)}: {base_name}")

print(f"Готово! Обработано изображений: {processed_count}")
print(f"Повреждённые изображения в: {DAMAGED_DIR}")
print(f"Маски сохранены в: {USED_MASK_DIR}")