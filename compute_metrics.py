# import os
# import cv2
# import numpy as np
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim
# from pathlib import Path

# # Пути (относительно src/)
# RESULTS_DIR = Path("../results/val_damaged_test")
# ORIGINAL_DIR = Path("../data/val/val_256")  # где лежат оригиналы (.jpg)
# DAMAGED_DIR = Path("../data/images/val_damaged")  # для проверки, если нужно
# PRED_SUFFIX = "_damaged_restored.png"

# psnr_list = []
# ssim_list = []

# # Соберём предсказания из RESULTS_DIR
# pred_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(PRED_SUFFIX)]

# for pred_file in pred_files:
#     base_name = pred_file.replace(PRED_SUFFIX, "")
#     orig_path = ORIGINAL_DIR / f"{base_name}.jpg"
#     pred_path = RESULTS_DIR / pred_file
    
#     if not orig_path.exists():
#         print(f"Не найден оригинал для {pred_file}: {orig_path}")
#         continue

#     orig = cv2.imread(str(orig_path))
#     pred = cv2.imread(str(pred_path))
    
#     if orig is None or pred is None:
#         print(f"Не удалось прочитать файлы для {pred_file}")
#         continue
    
#     orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
#     pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
    
#     # Обрезаем до минимального размера, если отличаются
#     h = min(orig.shape[0], pred.shape[0])
#     w = min(orig.shape[1], pred.shape[1])
#     orig = orig[:h, :w]
#     pred = pred[:h, :w]
    
#     p = psnr(orig, pred, data_range=255)
#     s = ssim(orig, pred, multichannel=True, data_range=255, channel_axis=-1)
    
#     psnr_list.append(p)
#     ssim_list.append(s)
    
#     print(f"{pred_file}: PSNR = {p:.2f} dB, SSIM = {s:.4f}")

# if psnr_list:
#     print(f"\nСреднее по {len(psnr_list)} изображениям:")
#     print(f"PSNR: {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f}")
#     print(f"SSIM: {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
# else:
#     print("Не найдено пар оригинал/предсказание — проверь пути и суффиксы")































import os
import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
from metric.metric import fid

try:
    from lpips import LPIPS
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("[WARNING] LPIPS not installed. Install with: pip install lpips")

# Пути (относительно src/)
RESULTS_DIR = Path("../results/val_finetuned")
ORIGINAL_DIR = Path("../data/images/val_orig_only")  # где лежат оригиналы (.jpg)
PRED_SUFFIX = "_damaged_restored.png"

# Инициализация устройства и моделей
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

if HAS_LPIPS:
    lpips_fn = LPIPS(net='vgg').to(device)
    lpips_fn.eval()

psnr_list = []
ssim_list = []
lpips_list = []
orig_images = []
pred_images = []

# Соберём оригиналы из ORIGINAL_DIR
orig_files = [f for f in os.listdir(ORIGINAL_DIR) if f.lower().endswith((".jpg", ".png"))]
print(f"[INFO] Found {len(orig_files)} original images")

matched_count = 0
for orig_file in sorted(orig_files):
    base_name = os.path.splitext(orig_file)[0]
    orig_path = ORIGINAL_DIR / orig_file
    pred_path = RESULTS_DIR / f"{base_name}{PRED_SUFFIX}"

    if not pred_path.exists():
        continue

    orig = cv2.imread(str(orig_path))
    pred = cv2.imread(str(pred_path))

    if orig is None or pred is None:
        print(f"[ERROR] Failed to read: {orig_file}")
        continue

    # BGR -> RGB
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

    # Resize to match
    h = min(orig.shape[0], pred.shape[0])
    w = min(orig.shape[1], pred.shape[1])
    orig_crop = orig[:h, :w]
    pred_crop = pred[:h, :w]

    # PSNR & SSIM
    p = psnr(orig_crop, pred_crop, data_range=255)
    s = ssim(orig_crop, pred_crop, multichannel=True, data_range=255, channel_axis=-1)
    psnr_list.append(p)
    ssim_list.append(s)

    # LPIPS
    if HAS_LPIPS:
        orig_tensor = torch.from_numpy(orig_crop.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        pred_tensor = torch.from_numpy(pred_crop.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        orig_tensor = orig_tensor.to(device)
        pred_tensor = pred_tensor.to(device)

        with torch.no_grad():
            lp = lpips_fn(orig_tensor, pred_tensor).item()
        lpips_list.append(lp)
        print(f"{orig_file}: PSNR = {p:.2f} dB, SSIM = {s:.4f}, LPIPS = {lp:.4f}")
    else:
        print(f"{orig_file}: PSNR = {p:.2f} dB, SSIM = {s:.4f}")

    # Accumulate for FID
    orig_images.append(orig_crop)
    pred_images.append(pred_crop)
    matched_count += 1

print(f"\n[INFO] Matched {matched_count} image pairs")

if psnr_list:
    print(f"\n{'='*60}")
    print(f"PSNR: {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} dB")
    print(f"SSIM: {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
    
    if lpips_list:
        print(f"LPIPS: {np.mean(lpips_list):.4f} ± {np.std(lpips_list):.4f}")

    # FID
    if len(orig_images) > 0:
        try:
            print(f"\n[INFO] Computing FID (this may take a while)...")
            fid_value = fid(orig_images, pred_images, num_worker=4)
            print(f"FID: {fid_value:.2f}")
        except Exception as e:
            print(f"[ERROR] FID computation failed: {e}")

    print(f"{'='*60}")
else:
    print("[ERROR] No matching pairs found!")
    print(f"RESULTS_DIR: {RESULTS_DIR}")
    print(f"ORIGINAL_DIR: {ORIGINAL_DIR}")
    print(f"PRED_SUFFIX: {PRED_SUFFIX}")

