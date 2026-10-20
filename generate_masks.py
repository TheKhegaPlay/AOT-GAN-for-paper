# import cv2
# import numpy as np
# import os
# import random

# NUM_MASKS = 1000  # сколько масок сгенерировать
# HEIGHT, WIDTH = 256, 256  # размер маски (как в твоём датасете)
# MIN_COVERAGE = 0.05  # минимальное покрытие
# MAX_COVERAGE = 0.30  # максимальное
# SAVE_DIR = "../data/masks/custom_masks"
# os.makedirs(SAVE_DIR, exist_ok=True)

# for i in range(NUM_MASKS):
#     mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    
#     # Сгенерируем случайные линии (царапины)
#     num_lines = random.randint(3, 15)
#     for _ in range(num_lines):
#         x1, y1 = random.randint(0, WIDTH), random.randint(0, HEIGHT)
#         x2, y2 = random.randint(0, WIDTH), random.randint(0, HEIGHT)
#         thickness = random.randint(3, 10)
#         cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
    
#     # Добавим блобы (круглые/овальные дыры)
#     num_blobs = random.randint(1, 5)
#     for _ in range(num_blobs):
#         center = (random.randint(0, WIDTH), random.randint(0, HEIGHT))
#         axes = (random.randint(10, 50), random.randint(10, 50))
#         angle = random.randint(0, 360)
#         cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)  # -1 = заполненный
    
#     # Проверим покрытие
#     coverage = np.sum(mask == 255) / (HEIGHT * WIDTH)
#     if coverage < MIN_COVERAGE or coverage > MAX_COVERAGE:
#         continue  # пропустим, если не в диапазоне
    
#     cv2.imwrite(os.path.join(SAVE_DIR, f"mask_{i:04d}.png"), mask)
#     print(f"Сгенерирована маска {i+1}/{NUM_MASKS}, покрытие {coverage:.2%}")

# print("Готово! Маски в ../data/masks/custom_masks")

import argparse
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def generate_single_blob_mask(
    h=256,
    w=256,
    min_coverage=0.04,
    max_coverage=0.12,
    max_attempts=100,
):
    if not 0 < min_coverage < max_coverage < 1:
        raise ValueError("coverage limits must satisfy 0 < min_coverage < max_coverage < 1")

    target_coverage = (min_coverage + max_coverage) / 2
    scale = float(np.clip(math.sqrt(target_coverage / 0.08), 0.7, 2.0))
    for attempt in range(max_attempts):
        canvas = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(canvas)

        # Одна непрерывная свободная кривая
        n_points = np.random.randint(8, 18)
        points = []
        # Стартовая точка ближе к центру, чтобы пятно не уходило за край
        cx = np.random.randint(w // 4, 3 * w // 4)
        cy = np.random.randint(h // 4, 3 * h // 4)
        points.append((cx, cy))

        for i in range(1, n_points):
            angle = np.random.uniform(0, 2 * np.pi)
            length_min = max(8, int(15 * scale))
            length_max = max(length_min + 1, int(45 * scale))
            length = np.random.randint(length_min, length_max)
            nx = int(np.clip(points[-1][0] + length * np.cos(angle), 10, w - 10))
            ny = int(np.clip(points[-1][1] + length * np.sin(angle), 10, h - 10))
            points.append((nx, ny))

        # Draw connected segments with variable thickness for scratch-like damage.
        for i in range(len(points) - 1):
            thickness_min = max(2, int(6 * scale))
            thickness_max = max(thickness_min + 1, int(16 * scale))
            thickness = np.random.randint(thickness_min, thickness_max)
            draw.line((points[i], points[i + 1]), fill=255, width=thickness)
            radius = thickness // 2
            for x, y in points[i:i + 2]:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)

        # Approximate a small morphological closing to fill tiny gaps.
        canvas = canvas.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
        mask = np.asarray(canvas, dtype=np.uint8)

        coverage = mask.mean() / 255.0
        if min_coverage <= coverage <= max_coverage:
            return mask

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate irregular single-blob masks")
    parser.add_argument("--num-masks", type=int, default=3000)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--min-coverage", type=float, default=0.04)
    parser.add_argument("--max-coverage", type=float, default=0.12)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attempts-per-mask", type=int, default=100)
    args = parser.parse_args()
    if args.num_masks <= 0 or args.height <= 0 or args.width <= 0:
        parser.error("num-masks, height and width must be positive")

    min_percent = round(args.min_coverage * 100)
    max_percent = round(args.max_coverage * 100)
    out_dir = args.output_dir or f"masks_single_blob_{min_percent}_{max_percent}"
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(args.seed)

    generated = 0
    attempts = 0
    max_total_attempts = args.num_masks * args.attempts_per_mask * 10
    while generated < args.num_masks:
        attempts += 1
        if attempts > max_total_attempts:
            raise RuntimeError(
                f"Generated {generated}/{args.num_masks} valid masks after "
                f"{max_total_attempts} attempts; widen the coverage interval"
            )
        mask = generate_single_blob_mask(
            h=args.height,
            w=args.width,
            min_coverage=args.min_coverage,
            max_coverage=args.max_coverage,
            max_attempts=args.attempts_per_mask,
        )
        if mask is None:
            continue
        output_path = os.path.join(out_dir, f"mask_{generated:05d}.png")
        Image.fromarray(mask, mode="L").save(output_path)
        generated += 1
        if generated % 100 == 0 or generated == args.num_masks:
            print(f"Generated {generated}/{args.num_masks} masks", flush=True)

    print(f"Generated {generated} masks in {out_dir}")