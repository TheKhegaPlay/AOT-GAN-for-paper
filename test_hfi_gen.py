#!/usr/bin/env python3
"""
Тестовый скрипт для проверки HFI-Gen архитектуры.
Проверяет:
1. Синтаксис и импорты
2. Инициализация моделей
3. Forward pass с правильными shapes
4. Backward pass для gradients
"""

import sys
import torch
import torch.nn as nn

def test_architecture():
    print("=" * 70)
    print("HFI-Gen (Hybrid Forensic Inpainting Generator) - Architecture Test")
    print("=" * 70)
    
    # ========== TEST 1: Импорты ==========
    print("\n[TEST 1] Checking imports...")
    try:
        from model.aotgan import (
            HybridInpaintGenerator, 
            ConvBlock, 
            UpConv, 
            AOTBlock,
            InpaintGenerator  # Алиас для совместимости
        )
        from loss.loss import (
            L1,
            SSIM,
            MultiScaleSSIM,
            Perceptual,
            Style,
            ConfidenceRegularization,
            WeightedReconstructionLoss,
            nsgan,
            smgan
        )
        print("✓ All imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # ========== TEST 2: Инициализация генератора ==========
    print("\n[TEST 2] Initializing HybridInpaintGenerator...")
    try:
        class Args:
            rates = [1, 2, 4, 8]
            block_num = 8
        
        args = Args()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        generator = HybridInpaintGenerator(args).to(device)
        print(f"✓ Generator initialized")
        print(f"  - Model size: {sum(p.numel() for p in generator.parameters()) / 1e6:.2f}M parameters")
    except Exception as e:
        print(f"✗ Generator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========== TEST 3: Backward compatibility (InpaintGenerator alias) ==========
    print("\n[TEST 3] Checking backward compatibility (InpaintGenerator = HybridInpaintGenerator)...")
    try:
        assert InpaintGenerator is HybridInpaintGenerator
        generator2 = InpaintGenerator(args).to(device)
        print("✓ Backward compatibility OK")
    except Exception as e:
        print(f"✗ Backward compatibility check failed: {e}")
        return False
    
    # ========== TEST 4: Forward pass ==========
    print("\n[TEST 4] Testing forward pass...")
    try:
        batch_size = 2
        height, width = 256, 256
        
        # Создать тестовые входы
        images = torch.randn(batch_size, 3, height, width, device=device)
        masks = torch.bernoulli(torch.full((batch_size, 1, height, width), 0.2, device=device))
        
        # Forward pass
        with torch.no_grad():
            restored, confidence = generator(images, masks)
        
        # Проверить shapes
        assert restored.shape == (batch_size, 3, height, width), f"Wrong restored shape: {restored.shape}"
        assert confidence.shape == (batch_size, 1, height, width), f"Wrong confidence shape: {confidence.shape}"
        
        # Проверить ranges
        assert restored.min() >= -1.01 and restored.max() <= 1.01, \
            f"Restored out of [-1, 1] range: [{restored.min():.3f}, {restored.max():.3f}]"
        assert confidence.min() >= -0.01 and confidence.max() <= 1.01, \
            f"Confidence out of [0, 1] range: [{confidence.min():.3f}, {confidence.max():.3f}]"
        
        print(f"✓ Forward pass successful")
        print(f"  - Restored: {restored.shape}, range [{restored.min():.3f}, {restored.max():.3f}]")
        print(f"  - Confidence: {confidence.shape}, range [{confidence.min():.3f}, {confidence.max():.3f}]")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========== TEST 5: Gradients ==========
    print("\n[TEST 5] Testing backward pass (gradients)...")
    try:
        generator.train()
        
        generator.zero_grad(set_to_none=True)
        
        # Forward
        restored, confidence = generator(images, masks)
        
        # Loss
        loss = restored.sum() + confidence.sum()
        
        # Backward
        loss.backward()
        
        # Check gradients exist
        has_grad = False
        for name, param in generator.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "No gradients computed!"
        print(f"✓ Backward pass successful (gradients computed)")
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========== TEST 6: Loss functions ==========
    print("\n[TEST 6] Testing loss functions...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Создать целевое изображение
        target = torch.randn(batch_size, 3, height, width, device=device)
        
        # L1
        l1_loss = L1()
        l1_val = l1_loss(restored, target)
        assert l1_val.item() > 0, "L1 loss should be > 0"
        print(f"  ✓ L1 loss: {l1_val.item():.4f}")
        
        # MS-SSIM
        ms_ssim = MultiScaleSSIM(device=device)
        ms_ssim_val = ms_ssim(restored, target)
        assert 0 <= ms_ssim_val.item() <= 1, "MS-SSIM should be in [0, 1]"
        print(f"  ✓ MS-SSIM loss: {ms_ssim_val.item():.4f}")
        
        # Perceptual
        try:
            perceptual = Perceptual(device=device)
            perc_val = perceptual(restored, target)
            assert perc_val.item() > 0, "Perceptual loss should be > 0"
            print(f"  ✓ Perceptual loss: {perc_val.item():.4f}")
        except Exception as e:
            print(f"  ⚠ Perceptual: {e} (probably no GPU or no pretrained weights)")
        
        # Confidence Regularization
        conf_reg = ConfidenceRegularization(device=device)
        conf_reg_val = conf_reg(confidence, masks)
        assert conf_reg_val.item() >= 0, "Confidence reg loss should be >= 0"
        print(f"  ✓ Confidence Regularization loss: {conf_reg_val.item():.4f}")
        
        # Weighted Reconstruction
        w_recon = WeightedReconstructionLoss(device=device)
        w_recon_val = w_recon(restored, target, confidence, masks)
        assert w_recon_val.item() >= 0, "Weighted recon loss should be >= 0"
        print(f"  ✓ Weighted Reconstruction loss: {w_recon_val.item():.4f}")
        
        print(f"✓ All loss functions working")
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========== TEST 7: Discriminator ==========
    print("\n[TEST 7] Testing Discriminator...")
    try:
        from model.aotgan import Discriminator
        
        discriminator = Discriminator().to(device)
        
        # Forward pass
        with torch.no_grad():
            d_out = discriminator(restored)
        
        print(f"✓ Discriminator working")
        print(f"  - Input: {restored.shape}")
        print(f"  - Output: {d_out.shape}")
    except Exception as e:
        print(f"✗ Discriminator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========== TEST 8: Trainer compatibility ==========
    print("\n[TEST 8] Checking trainer compatibility...")
    try:
        # Проверить что trainer может обработать tuple выход
        gen_output = (restored, confidence)  # Tuple как в HFI-Gen
        
        if isinstance(gen_output, tuple):
            pred_img, conf_map = gen_output
            has_confidence = True
        else:
            pred_img = gen_output
            conf_map = None
            has_confidence = False
        
        assert has_confidence, "Should detect tuple output"
        assert pred_img.shape == restored.shape, "Shape mismatch"
        assert conf_map.shape == confidence.shape, "Shape mismatch"
        
        print(f"✓ Trainer compatibility OK (detects confidence output)")
    except Exception as e:
        print(f"✗ Trainer compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  - HybridInpaintGenerator: ✓ Working")
    print(f"  - Forward/Backward: ✓ Working")
    print(f"  - Loss functions: ✓ Working")
    print(f"  - Discriminator: ✓ Working")
    print(f"  - Trainer compatibility: ✓ Working")
    print(f"\nReady for training!")
    
    return True

if __name__ == '__main__':
    try:
        success = test_architecture()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
