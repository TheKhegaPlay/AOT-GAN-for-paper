# HFI-Gen (Hybrid Forensic Inpainting Generator) — Руководство по использованию

## Что изменилось?

Переход с базового **AOT-GAN** на **HFI-Gen** (Hybrid Forensic Inpainting Generator):

### Основные изменения

| Компонент | Было | Стало | Результат |
|-----------|------|-------|-----------|
| **Генератор** | `InpaintGenerator` (3 уровня) | `HybridInpaintGenerator` (4 уровня + confidence branch) | Лучше сохраняет детали, честная уверенность |
| **Выход генератора** | `image: (B, 3, H, W)` | `(image, confidence): ((B, 3, H, W), (B, 1, H, W))` | Видимость уверенности модели |
| **Loss function** | `L1 + Adversarial` | `L1 + MS-SSIM + Perceptual + Style + Adversarial + ConfidenceReg + WeightedRecon` | Высокое качество + честность |
| **Skip-connections** | Нет | Да (e1, e2, e3 → decoder) | Четче текст, сохранены грани |
| **Trainer compatibility** | N/A | Backward compatible | Работает как с новой, так и со старой архитектурой |

---

## Быстрый старт

### 1. Базовое обучение (новая HFI-Gen архитектура)

```bash
python train.py \
    --model aotgan \
    --data_train places365_standard \
    --data_test div2k \
    --mask_type irregular \
    --batch_size 8 \
    --iterations 500000 \
    --lrg 0.001 \
    --lrd 0.0001
```

**Что происходит:**
- Trainer автоматически обнаружит, что генератор возвращает кортеж `(image, confidence)`
- Инициализирует confidence-aware losses
- Обучает модель с использованием всех компонентов loss function

### 2. Обучение с явной конфигурацией confidence weights

Добавьте в вашу конфиг опции:

```python
# В utils/option.py или через аргументы командной строки
parser.add_argument('--conf_reg_weight', type=float, default=0.3,
    help='Weight для confidence regularization loss')
parser.add_argument('--w_recon_weight', type=float, default=0.2,
    help='Weight для weighted reconstruction loss')
```

Затем:

```bash
python train.py \
    --model aotgan \
    --conf_reg_weight 0.3 \
    --w_recon_weight 0.2 \
    [другие аргументы]
```

### 3. Инференс (тестирование) с выводом confidence map

```python
import torch
from model.aotgan import HybridInpaintGenerator
from PIL import Image
import torchvision.transforms as transforms

# Загрузить модель
model = HybridInpaintGenerator(args)
model.load_state_dict(torch.load('path/to/checkpoint.pt'))
model.eval()

# Подготовить входные данные
image = Image.open('damaged_image.jpg')  # (H, W, 3)
mask = Image.open('damage_mask.png')     # (H, W, 1) - 1 где повреждено

# Нормализовать
img_tensor = transforms.ToTensor(image) * 2 - 1  # [-1, 1]
mask_tensor = transforms.ToTensor(mask)           # [0, 1]

# Добавить batch dimension
img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)
mask_tensor = mask_tensor.unsqueeze(0)  # (1, 1, H, W)

# Инференс
with torch.no_grad():
    restored_img, confidence_map = model(img_tensor, mask_tensor)

# restored_img: (1, 3, H, W) в [-1, 1]
# confidence_map: (1, 1, H, W) в [0, 1]

# Денормализовать и сохранить
restored_img = (restored_img[0] + 1) / 2  # [0, 1]
confidence_map = confidence_map[0]

# Сохранить результаты
restored_pil = transforms.ToPILImage()(restored_img.cpu())
restored_pil.save('restored_image.jpg')

confidence_pil = transforms.ToPILImage()(confidence_map.cpu())
confidence_pil.save('confidence_map.png')
```

---

## Использование Confidence Map для судебной оценки

### 1. Базовая интерпретация

```python
# После инференса
confidence_map = confidence_map.squeeze()  # (H, W)

# Статистика уверенности
print(f"Средняя уверенность: {confidence_map.mean():.3f}")
print(f"Мин. уверенность: {confidence_map.min():.3f}")
print(f"Макс. уверенность: {confidence_map.max():.3f}")

# Процент регионов с высокой уверенностью
high_confidence = (confidence_map > 0.8).float().mean()
print(f"Регионов с confidence > 0.8: {high_confidence*100:.1f}%")
```

### 2. Визуализация confidence-aware восстановления

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Оригинальное изображение с маской
ax = axes[0, 0]
ax.imshow((img_tensor[0].cpu().permute(1, 2, 0) + 1) / 2)
ax.set_title('Original Image')
ax.axis('off')

# Восстановленное изображение
ax = axes[0, 1]
ax.imshow((restored_img.cpu().permute(1, 2, 0)))
ax.set_title('Restored Image')
ax.axis('off')

# Карта уверенности (heat map)
ax = axes[1, 0]
im = ax.imshow(confidence_map.cpu(), cmap='hot', vmin=0, vmax=1)
ax.set_title('Confidence Map (Heat Map)')
plt.colorbar(im, ax=ax)
ax.axis('off')

# Маска повреждения
ax = axes[1, 1]
ax.imshow(mask_tensor[0, 0].cpu(), cmap='gray')
ax.set_title('Damage Mask')
ax.axis('off')

plt.tight_layout()
plt.savefig('forensic_analysis.png', dpi=150)
plt.show()
```

### 3. Судебный отчёт: Пороги уверенности

```python
def generate_forensic_report(confidence_map, mask, image_path):
    """Создать судебный отчёт с анализом надёжности восстановления."""
    
    confidence = confidence_map.squeeze().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    
    # Анализ только в области маски (восстановленные регионы)
    restored_confidence = confidence[mask > 0.5]
    
    report = f"""
    ═══════════════════════════════════════════════════════════════
    FORENSIC IMAGE RESTORATION REPORT
    Image: {image_path}
    ═══════════════════════════════════════════════════════════════
    
    CONFIDENCE STATISTICS (в области восстановления):
    ───────────────────────────────────────────────────
    Mean Confidence:     {restored_confidence.mean():.3f}
    Std. Deviation:      {restored_confidence.std():.3f}
    Min. Confidence:     {restored_confidence.min():.3f}
    Max. Confidence:     {restored_confidence.max():.3f}
    
    ADMISSIBILITY ASSESSMENT:
    ────────────────────────────────
    """
    
    # Пороги для судебного использования
    very_high = (restored_confidence > 0.9).sum() / len(restored_confidence)
    high = ((restored_confidence > 0.75) & (restored_confidence <= 0.9)).sum() / len(restored_confidence)
    medium = ((restored_confidence > 0.5) & (restored_confidence <= 0.75)).sum() / len(restored_confidence)
    low = (restored_confidence <= 0.5).sum() / len(restored_confidence)
    
    report += f"""
    Confidence > 0.90 (VERY HIGH):  {very_high*100:5.1f}%  ✓ Допустимо как доказательство
    Confidence 0.75-0.90 (HIGH):    {high*100:5.1f}%  ✓ Допустимо с пояснением эксперта
    Confidence 0.50-0.75 (MEDIUM):  {medium*100:5.1f}%  ⚠ Требует тщательного анализа
    Confidence < 0.50 (LOW):        {low*100:5.1f}%  ✗ Не рекомендуется использовать
    
    RECOMMENDATION:
    ───────────────
    """
    
    if very_high > 0.7:
        report += "✓ Восстановление достаточно надёжно для использования в суде.\n"
    elif high + very_high > 0.5:
        report += "⚠ Восстановление может быть использовано с оговорками эксперта.\n"
    else:
        report += "✗ Восстановление НЕ рекомендуется использовать как доказательство.\n"
    
    report += "═══════════════════════════════════════════════════════════════\n"
    
    return report

# Использование
report = generate_forensic_report(confidence_map, mask_tensor, "image.jpg")
print(report)
```

---

## Backward Compatibility (совместимость с оригинальным AOT-GAN)

### Использование старого кода

Если у вас есть старый код, который ожидает одного выхода от генератора:

```python
# Старый код (до HFI-Gen)
pred_img = model(images, masks)  # ожидает (B, 3, H, W)
```

**Новый код (HFI-Gen)** поддерживает ОБОЕ варианты:

```python
# Вариант 1: новая архитектура с confidence
restored, confidence = model(images, masks)

# Вариант 2: совместимость со старым кодом
gen_output = model(images, masks)
if isinstance(gen_output, tuple):
    restored, confidence = gen_output
else:
    restored = gen_output
```

### Trainer поддерживает оба режима

Trainer в `trainer/trainer.py` **автоматически обнаружит** тип выхода генератора:

```python
gen_output = self.netG(images_masked, masks)

# Trainer проверяет тип выхода
if isinstance(gen_output, tuple):
    pred_img, confidence_map = gen_output
    has_confidence = True  # Используются confidence losses
else:
    pred_img = gen_output
    confidence_map = None
    has_confidence = False  # Используются стандартные losses
```

---

## Кастомизация Loss Weights

### Стандартная конфигурация (рекомендуемая)

```python
# В training config
losses_weights = {
    'L1':           1.0,      # Пиксельная точность
    'MSssim':       0.5,      # Структурное сходство на разных масштабах
    'Perceptual':   0.1,      # Высокоуровневые фичи (VGG)
    'Style':        0.05,     # Текстурное сходство
    'adversarial':  0.1,      # Натуральность (discriminator)
}

confidence_losses = {
    'conf_reg_weight':  0.3,  # Confidence regularization
    'w_recon_weight':   0.2,  # Weighted reconstruction
}
```

### Для "soft" восстановления (когда неопределённость важнее точности)

Используйте, если повреждение серьёзное и точное восстановление невозможно:

```python
losses_weights = {
    'L1':           0.5,      # ↓ Меньше требуемная точность
    'MSssim':       0.3,      # ↓
    'Perceptual':   0.15,     # ↑ Больше внимание на семантику
    'Style':        0.10,     # ↑
    'adversarial':  0.1,
}

confidence_losses = {
    'conf_reg_weight':  0.5,  # ↑ Более честная оценка уверенности
    'w_recon_weight':   0.3,  # ↑
}
```

### Для "hard" восстановления (когда нужна максимальная точность)

Используйте для минимальных повреждений:

```python
losses_weights = {
    'L1':           1.5,      # ↑ Высокая требуемая точность
    'MSssim':       0.8,      # ↑
    'Perceptual':   0.05,     # ↓
    'Style':        0.02,     # ↓
    'adversarial':  0.1,
}

confidence_losses = {
    'conf_reg_weight':  0.2,  # ↓ Модель уверена в точности
    'w_recon_weight':   0.15, # ↓
}
```

---

## Диагностика проблем

### Проблема: "ModuleNotFoundError: No module named 'MultiScaleSSIM'"

**Решение**: Убедитесь, что вы используете обновленный `loss/loss.py`. Проверьте, что все новые классы инициализированы:

```python
from loss.loss import (
    L1, 
    SSIM, MultiScaleSSIM,  # ← новые
    Perceptual, Style, 
    ConfidenceRegularization, WeightedReconstructionLoss,  # ← новые
    nsgan, smgan
)
```

### Проблема: "TypeError: forward() got an unexpected keyword argument"

**Решение**: Убедитесь, что loss функция инициализирована с правильными параметрами:

```python
# ✗ Неправильно
loss = SSIM()  # нет device!

# ✓ Правильно  
loss = SSIM(device='cuda')
```

### Проблема: Confidence map всегда около 0.5 (не обучается)

**Возможные причины:**
1. **Loss weight слишком низкий**: Увеличьте `conf_reg_weight` до 0.5-1.0
2. **Недостаточно итераций**: Confidence branch нужно дольше обучаться (хотя бы 50K итераций)
3. **Плохой data**: Убедитесь, что маски разные (не всегда одинаковые повреждения)

**Решение:**

```bash
python train.py \
    --model aotgan \
    --iterations 500000 \
    --conf_reg_weight 0.5 \
    --w_recon_weight 0.3 \
    --print_every 100  # Чаще смотрите на confidence loss значения
```

---

## Полезные команды для тестирования

### Тестирование на одном батче

```python
# test_hfi_gen.py
import torch
from model.aotgan import HybridInpaintGenerator
from utils.option import args

# Создать модель
model = HybridInpaintGenerator(args)
model.eval()

# Тестовые данные
batch_size = 4
height, width = 256, 256
images = torch.randn(batch_size, 3, height, width)
masks = torch.bernoulli(torch.full((batch_size, 1, height, width), 0.2))

# Инференс
with torch.no_grad():
    restored, confidence = model(images, masks)

# Проверить выходы
print(f"Restored shape: {restored.shape}")
print(f"Confidence shape: {confidence.shape}")
print(f"Restored range: [{restored.min():.3f}, {restored.max():.3f}]")
print(f"Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")

assert restored.shape == (batch_size, 3, height, width)
assert confidence.shape == (batch_size, 1, height, width)
assert restored.min() >= -1.01 and restored.max() <= 1.01
assert confidence.min() >= -0.01 and confidence.max() <= 1.01

print("✓ All tests passed!")
```

---

## Ссылки и документация

- [Архитектурное обоснование](./ARCHITECTURE_RATIONALE.md) — подробное объяснение всех решений
- [Model code](./model/aotgan.py) — код генератора и loss функций
- [Loss code](./loss/loss.py) — реализация всех loss компонентов
- [Trainer code](./trainer/trainer.py) — тренировочный цикл
