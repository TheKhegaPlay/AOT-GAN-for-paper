import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import VGG19, GaussianBlur


class L1:
    def __init__(self):
        self.calc = torch.nn.L1Loss()

    def __call__(self, x, y):
        return self.calc(x, y)


class SSIM(nn.Module):
    """
    Structural Similarity Index (SSIM).
    Измеряет структурное сходство между изображениями.
    Для судебных изображений важна структурная целостность (текст, грани, объекты).
    """
    def __init__(self, window_size=11, sigma=1.5, device='cuda'):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.device = device

        # Гауссовское окно
        coordinates = torch.arange(window_size, dtype=torch.float32)
        gauss = torch.exp(
            -((coordinates - window_size // 2) ** 2) / float(2 * sigma ** 2)
        )
        kernel = gauss / gauss.sum()
        self.register_buffer('kernel1d', kernel.view(1, 1, -1).to(device))

    def _ssim(self, x, y):
        """SSIM между двумя изображениями (одноканальными)."""
        b, c, h, w = x.shape
        
        # Гауссово сглаживание
        kernel = self.kernel1d.reshape(-1)
        kernel2d = torch.outer(kernel, kernel).view(
            1, 1, self.window_size, self.window_size
        )
        kernel2d = kernel2d.repeat(c, 1, 1, 1).to(x.device)
        
        padding = self.window_size // 2
        
        mu_x = F.conv2d(x, kernel2d, padding=padding, groups=c)
        mu_y = F.conv2d(y, kernel2d, padding=padding, groups=c)
        
        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x2 = F.conv2d(x ** 2, kernel2d, padding=padding, groups=c) - mu_x2
        sigma_y2 = F.conv2d(y ** 2, kernel2d, padding=padding, groups=c) - mu_y2
        sigma_xy = F.conv2d(x * y, kernel2d, padding=padding, groups=c) - mu_xy
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
        
        return ssim_map.mean()

    def forward(self, x, y):
        """Вычисляет SSIM для RGB изображений."""
        return self._ssim(x, y)


class MultiScaleSSIM(nn.Module):
    """
    Multi-Scale SSIM (MS-SSIM).
    Вычисляет SSIM на разных масштабах для учёта как локальных, так и глобальных деталей.
    Особенно важно для forensic восстановления: локальные детали (следы, шумы) 
    и глобальная структура (объекты, грани) должны быть восстановлены корректно.
    """
    def __init__(self, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device='cuda'):
        super(MultiScaleSSIM, self).__init__()
        self.weights = torch.tensor(weights, device=device)
        self.ssim = SSIM(device=device)

    def forward(self, x, y):
        """
        Args:
            x, y: (B, 3, H, W) в диапазоне [-1, 1]
        Returns:
            1 - ms_ssim (loss)
        """
        # Нормализуем в [0, 1] для SSIM
        x_norm = (x + 1.0) / 2.0
        y_norm = (y + 1.0) / 2.0
        
        ms_ssim = 0.0
        for i, w in enumerate(self.weights):
            ssim_val = self.ssim(x_norm, y_norm)
            ms_ssim += w * ssim_val
            
            # Downsample для следующего масштаба (кроме последнего)
            if i < len(self.weights) - 1:
                x_norm = F.avg_pool2d(x_norm, kernel_size=2, stride=2)
                y_norm = F.avg_pool2d(y_norm, kernel_size=2, stride=2)
        
        return 1.0 - ms_ssim




class Perceptual(nn.Module):
    """
    Perceptual Loss (LPIPS-style).
    Использует VGG19 для сравнения высокоуровневых фич.
    Сохраняет семантическое содержание восстановленного изображения.
    """
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0], device='cuda'):
        super(Perceptual, self).__init__()
        self.vgg = VGG19(device=device).to(device)
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        with torch.no_grad():
            x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        
        content_loss = 0.0
        prefix = [1, 2, 3, 4, 5]
        for i in range(5):
            content_loss += self.weights[i] * self.criterion(
                x_vgg[f"relu{prefix[i]}_1"], 
                y_vgg[f"relu{prefix[i]}_1"]
            )
        return content_loss


class Style(nn.Module):
    """
    Style Loss.
    Сравнивает стилистические характеристики через Gram matrices.
    Помогает восстановленной области соответствовать текстурным характеристикам.
    """
    def __init__(self, device='cuda'):
        super(Style, self).__init__()
        self.vgg = VGG19(device=device).to(device)
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * c)
        return G

    def __call__(self, x, y):
        with torch.no_grad():
            x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        
        style_loss = 0.0
        prefix = [2, 3, 4, 5]
        posfix = [2, 4, 4, 2]
        for pre, pos in list(zip(prefix, posfix)):
            style_loss += self.criterion(
                self.compute_gram(x_vgg[f"relu{pre}_{pos}"]), 
                self.compute_gram(y_vgg[f"relu{pre}_{pos}"])
            )
        return style_loss


class ConfidenceRegularization(nn.Module):
    """
    Confidence-Aware Regularization.
    
    Для судебной экспертизы важно:
    1. Штрафовать низкую уверенность внутри поврежденной области (маски)
    2. Взвешивать reconstruction loss на основе confidence map
    3. Избегать ложной самоуверенности модели
    
    Стратегия:
    - Область маски должна иметь HIGH confidence если восстановление хорошее
    - Регионы вне маски должны иметь HIGH confidence (неповреждённые)
    - Штрафуем низкую confidence если пиксель в маске (должен быть восстановлен уверенно)
    """
    def __init__(self, device='cuda', weight_valid=1.0, weight_masked=0.5):
        super(ConfidenceRegularization, self).__init__()
        self.device = device
        self.weight_valid = weight_valid
        self.weight_masked = weight_masked

    def forward(self, confidence_map, mask):
        """
        Args:
            confidence_map: (B, 1, H, W) - модель-оценка уверенности [0, 1]
            mask: (B, 1, H, W) - маска повреждения (1 = поврежденный, 0 = целый)
        
        Returns:
            loss - среднее значение confidence regularization
        
        Идея:
        - В неповреждённых областях (mask=0) confidence должна быть высокая
        - В поврежденных областях (mask=1) штрафуем низкую confidence
        """
        # Инвертируем маску: valid_mask = 1 где нет повреждений, 0 где есть
        valid_mask = 1.0 - mask
        
        # Для неповреждённых пиксельных штрафуем низкую confidence
        loss_valid = (1.0 - confidence_map) * valid_mask
        
        # Для поврежденных пиксельных штрафуем низкую confidence (должна быть уверена в восстановлении)
        loss_masked = (1.0 - confidence_map) * mask
        
        total_loss = (
            self.weight_valid * loss_valid.mean() + 
            self.weight_masked * loss_masked.mean()
        )
        
        return total_loss


class WeightedReconstructionLoss(nn.Module):
    """
    Weighted Reconstruction Loss.
    
    Взвешивает reconstruction loss на основе confidence map.
    Регионы с высокой уверенностью модели должны быть более точными.
    Это побуждает модель быть честной в своей уверенности.
    """
    def __init__(self, alpha=0.5, device='cuda'):
        super(WeightedReconstructionLoss, self).__init__()
        self.alpha = alpha  # Сила взвешивания
        self.criterion = nn.L1Loss(reduction='none')
        self.device = device

    def forward(self, restored, target, confidence_map, mask):
        """
        Args:
            restored: (B, 3, H, W) - восстановленное изображение
            target: (B, 3, H, W) - целевое изображение
            confidence_map: (B, 1, H, W) - карта уверенности [0, 1]
            mask: (B, 1, H, W) - маска повреждения (1 = поврежденный)
        
        Returns:
            loss - взвешенный reconstruction loss
        
        Применяем loss только в области маски (так как это восстановленная область)
        Взвешиваем на confidence: w = 1 + alpha * confidence
        """
        recon_loss = self.criterion(restored, target)  # (B, 3, H, W)
        
        # Получаем средний confidence для каждого пикселя
        confidence_expanded = confidence_map.repeat(1, 3, 1, 1)  # (B, 3, H, W)
        
        # Вес: чем выше confidence, тем выше штраф за ошибку
        weight = 1.0 + self.alpha * confidence_expanded
        
        # Применяем вес только в области маски
        weighted_loss = recon_loss * weight * mask.repeat(1, 3, 1, 1)
        
        return weighted_loss.mean()





class nsgan:
    """Non-saturating GAN loss (с Softplus)."""
    def __init__(self):
        self.loss_fn = torch.nn.Softplus()

    def __call__(self, netD, fake, real):
        """
        Args:
            netD: discriminator
            fake: (B, 3, H, W) - восстановленное изображение
            real: (B, 3, H, W) - целевое изображение
        """
        fake_detach = fake.detach()
        d_fake = netD(fake_detach)
        d_real = netD(real)
        dis_loss = self.loss_fn(-d_real).mean() + self.loss_fn(d_fake).mean()

        g_fake = netD(fake)
        gen_loss = self.loss_fn(-g_fake).mean()

        return dis_loss, gen_loss


class smgan:
    """
    Semantic GAN loss (с Gaussian blur на маске).
    Использует маску для выделения восстановленной области.
    """
    def __init__(self, ksize=71, device='cuda'):
        self.ksize = ksize
        self.loss_fn = nn.MSELoss()
        self.device = device
        self.gaussian_blur = GaussianBlur(
            (ksize, ksize),
            (10, 10),
            device=device
        )

    def __call__(self, netD, fake, real, masks):
        """
        Args:
            netD: discriminator
            fake: (B, 3, H, W) - восстановленное
            real: (B, 3, H, W) - целевое
            masks: (B, 1, H, W) - маска повреждения
        """
        fake_detach = fake.detach()

        g_fake = netD(fake)
        d_fake = netD(fake_detach)
        d_real = netD(real)

        _, _, h, w = g_fake.size()
        b, c, ht, wt = masks.size()

        # Выравниваем размеры если нужно
        if h != ht or w != wt:
            g_fake = F.interpolate(g_fake, size=(ht, wt), mode="bilinear", align_corners=True)
            d_fake = F.interpolate(d_fake, size=(ht, wt), mode="bilinear", align_corners=True)
            d_real = F.interpolate(d_real, size=(ht, wt), mode="bilinear", align_corners=True)
        
        d_fake_label = self.gaussian_blur(masks).detach()
        d_real_label = torch.zeros_like(d_real).to(masks.device)
        g_fake_label = torch.ones_like(g_fake).to(masks.device)

        dis_loss = self.loss_fn(d_fake, d_fake_label) + self.loss_fn(d_real, d_real_label)
        gen_loss = self.loss_fn(g_fake, g_fake_label) * masks / torch.mean(masks)

        return dis_loss.mean(), gen_loss.mean()
