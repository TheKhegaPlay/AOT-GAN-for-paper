# import importlib
# import os
# from glob import glob

# import numpy as np
# import torch
# from PIL import Image
# from torchvision.transforms import ToTensor
# from utils.option import args


# def postprocess(image):
#     image = torch.clamp(image, -1.0, 1.0)
#     image = (image + 1) / 2.0 * 255.0
#     image = image.permute(1, 2, 0)
#     image = image.cpu().numpy().astype(np.uint8)
#     return Image.fromarray(image)


# def main_worker(args):
#     net = importlib.import_module("model." + args.model)
#     model = net.InpaintGenerator(args).cuda()
#     model.load_state_dict(torch.load(args.pre_train, map_location="cuda"))
#     model.eval()

#     image_paths = sorted(glob(os.path.join(args.dir_image, "*.jpg")) +
#                          glob(os.path.join(args.dir_image, "*.png")))
#     mask_paths = sorted(glob(os.path.join(args.dir_mask, "*.png")))

#     os.makedirs(args.outputs, exist_ok=True)

#     for ipath, mpath in zip(image_paths, mask_paths):
#         # --- load image ---
#         image = ToTensor()(Image.open(ipath).convert("RGB"))
#         image = image * 2.0 - 1.0
#         image = image.unsqueeze(0).cuda()

#         # --- load mask ---
#         mask = ToTensor()(Image.open(mpath).convert("L"))
#         mask = (mask > 0.5).float()   # строго бинарная
#         mask = mask.unsqueeze(0).cuda()

#         # --- masked input ---
#         image_masked = image * (1 - mask)

#         with torch.no_grad():
#             pred = model(image_masked, mask)

#         # --- final restored image ---
#         restored = image * (1 - mask) + pred * mask

#         name = os.path.splitext(os.path.basename(ipath))[0]

#         postprocess(image_masked[0]).save(
#             os.path.join(args.outputs, f"{name}_masked.png")
#         )
#         postprocess(pred[0]).save(
#             os.path.join(args.outputs, f"{name}_pred.png")
#         )
#         postprocess(restored[0]).save(
#             os.path.join(args.outputs, f"{name}_restored.png")
#         )

#         print(f"Saved {name}")



# import importlib
# import os
# from glob import glob

# import numpy as np
# import torch
# from PIL import Image
# from torchvision.transforms import ToTensor
# from utils.option import args


# def postprocess(image):
#     image = torch.clamp(image, -1.0, 1.0)
#     image = (image + 1) / 2.0 * 255.0
#     image = image.permute(1, 2, 0)
#     image = image.cpu().numpy().astype(np.uint8)
#     return Image.fromarray(image)


# def main_worker(args, use_gpu=True):
#     # device = torch.device("cuda") if use_gpu else torch.device("cpu")

#     # Model and version
#     net = importlib.import_module("model." + args.model)
#     model = net.InpaintGenerator(args).cuda()
#     model.load_state_dict(torch.load(args.pre_train, map_location="cuda"))
#     model.eval()

#     # prepare dataset
#     image_paths = []
#     for ext in [".jpg", ".png"]:
#         image_paths.extend(glob(os.path.join(args.dir_image, "*" + ext)))
#     image_paths.sort()
#     mask_paths = sorted(glob(os.path.join(args.dir_mask, "*.png")))
#     os.makedirs(args.outputs, exist_ok=True)

#     # iteration through datasets
#     for ipath, mpath in zip(image_paths, mask_paths):
#         image = ToTensor()(Image.open(ipath).convert("RGB"))
#         image = (image * 2.0 - 1.0).unsqueeze(0)
#         mask = ToTensor()(Image.open(mpath).convert("L"))
#         mask = mask.unsqueeze(0)
#         image, mask = image.cuda(), mask.cuda()
#         image_masked = image * (1 - mask.float()) + mask

#         with torch.no_grad():
#             pred_img = model(image_masked, mask)

#         # comp_imgs = (1 - mask) * image + mask * pred_img
#         image_name = os.path.basename(ipath).split(".")[0]
#         postprocess(image_masked[0]).save(os.path.join(args.outputs, f"{image_name}_masked.png"))
#         postprocess(pred_img[0]).save(os.path.join(args.outputs, f"{image_name}_pred.png"))
#         # postprocess(comp_imgs[0]).save(os.path.join(args.outputs, f"{image_name}_comp.png"))
#         print(f"saving to {os.path.join(args.outputs, image_name)}")


# if __name__ == "__main__":
#     main_worker(args)











# import importlib
# import os
# from glob import glob

# import numpy as np
# import torch
# from PIL import Image
# from torchvision.transforms import ToTensor
# from utils.option import args


# def postprocess(image):
#     image = torch.clamp(image, -1.0, 1.0)
#     image = (image + 1) / 2.0 * 255.0
#     image = image.permute(1, 2, 0)
#     image = image.cpu().numpy().astype(np.uint8)
#     return Image.fromarray(image)


# def main_worker(args):
#     net = importlib.import_module("model." + args.model)
#     model = net.InpaintGenerator(args).cuda()
#     model.load_state_dict(torch.load(args.pre_train, map_location="cuda"))
#     model.eval()

#     image_paths = sorted(glob(os.path.join(args.dir_image, "*.jpg")) +
#                          glob(os.path.join(args.dir_image, "*.png")))
#     mask_paths = sorted(glob(os.path.join(args.dir_mask, "*.png")))

#     os.makedirs(args.outputs, exist_ok=True)

#     print(f"Images: {len(image_paths)}")
#     print(f"Masks:  {len(mask_paths)}")


#     for ipath, mpath in zip(image_paths, mask_paths):
#         # --- load image ---
#         image = ToTensor()(Image.open(ipath).convert("RGB"))
#         image = image * 2.0 - 1.0
#         image = image.unsqueeze(0).cuda()

#         # --- load mask ---
#         mask = ToTensor()(Image.open(mpath).convert("L"))
#         mask = (mask > 0.5).float()   # строго бинарная
#         mask = mask.unsqueeze(0).cuda()

#         # --- masked input ---
#         image_masked = image * (1 - mask)

#         with torch.no_grad():
#             pred = model(image_masked, mask)

#         # --- final restored image ---
#         restored = image * (1 - mask) + pred * mask

#         name = os.path.splitext(os.path.basename(ipath))[0]

#         postprocess(image_masked[0]).save(
#             os.path.join(args.outputs, f"{name}_masked.png")
#         )
#         postprocess(pred[0]).save(
#             os.path.join(args.outputs, f"{name}_pred.png")
#         )
#         postprocess(restored[0]).save(
#             os.path.join(args.outputs, f"{name}_restored.png")
#         )

#         print(f"Saved {name}")

























# import importlib
# import os
# from glob import glob

# import numpy as np
# import torch
# from PIL import Image
# from torchvision.transforms import ToTensor
# from utils.option import args


# def postprocess(image):
#     image = torch.clamp(image, -1.0, 1.0)
#     image = (image + 1) / 2.0 * 255.0
#     image = image.permute(1, 2, 0)
#     image = image.cpu().numpy().astype(np.uint8)
#     return Image.fromarray(image)


# def main_worker(args):
#     # Загрузка модели
#     net = importlib.import_module("model." + args.model)
#     model = net.InpaintGenerator(args).cuda()
#     model.load_state_dict(torch.load(args.pre_train, map_location="cuda"))
#     model.eval()

#     # Сбор путей к изображениям и маскам
#     image_paths = sorted(glob(os.path.join(args.dir_image, "*.jpg")) +
#                          glob(os.path.join(args.dir_image, "*.png")))
#     mask_paths = sorted(glob(os.path.join(args.dir_mask, "*.png")))

#     if len(image_paths) != len(mask_paths):
#         print(f"Несоответствие количества изображений ({len(image_paths)}) и масок ({len(mask_paths)})")
#         return

#     os.makedirs(args.outputs, exist_ok=True)

#     for ipath, mpath in zip(image_paths, mask_paths):
#         # Загрузка изображения
#         image = ToTensor()(Image.open(ipath).convert("RGB"))
#         image = (image * 2.0 - 1.0).unsqueeze(0).cuda()

#         # Загрузка маски (строго бинарная)
#         mask = ToTensor()(Image.open(mpath).convert("L"))
#         mask = (mask > 0.5).float().unsqueeze(0).cuda()

#         # Повреждённое изображение
#         image_masked = image * (1 - mask)

#         with torch.no_grad():
#             pred = model(image_masked, mask)

#         # Финальное восстановленное изображение (композиция)
#         restored = image * (1 - mask) + pred * mask

#         name = os.path.splitext(os.path.basename(ipath))[0]

#         # Сохранение только необходимого
#         postprocess(image_masked[0]).save(os.path.join(args.outputs, f"{name}_masked.png"))
#         postprocess(pred[0]).save(os.path.join(args.outputs, f"{name}_pred_raw.png"))  # сырой предсказанный
#         postprocess(restored[0]).save(os.path.join(args.outputs, f"{name}_restored.png"))  # финальное чистое

#         print(f"Обработано и сохранено: {name}")

#     print(f"Все результаты сохранены в {args.outputs}")


# if __name__ == "__main__":
#     main_worker(args)





















# import importlib
# import os
# from glob import glob
# from itertools import cycle

# import torch
# import numpy as np
# from PIL import Image
# from torchvision.transforms import ToTensor

# from utils.option import args


# # -------------------------
# # Utils
# # -------------------------
# def postprocess(image: torch.Tensor) -> Image.Image:
#     """
#     [-1, 1] tensor -> uint8 PIL Image
#     """
#     image = torch.clamp(image, -1.0, 1.0)
#     image = (image + 1.0) / 2.0 * 255.0
#     image = image.permute(1, 2, 0)
#     image = image.cpu().numpy().astype(np.uint8)
#     return Image.fromarray(image)


# def collect_images(root):
#     return sorted(
#         glob(os.path.join(root, "**", "*.jpg"), recursive=True) +
#         glob(os.path.join(root, "**", "*.png"), recursive=True) +
#         glob(os.path.join(root, "**", "*.JPG"), recursive=True) +
#         glob(os.path.join(root, "**", "*.PNG"), recursive=True)
#     )


# # -------------------------
# # Main
# # -------------------------
# def main_worker(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # --- load model ---
#     net = importlib.import_module("model." + args.model)
#     model = net.InpaintGenerator(args).to(device)
#     model.load_state_dict(torch.load(args.pre_train, map_location=device))
#     model.eval()

#     # --- collect data ---
#     image_paths = collect_images(args.dir_image)
#     mask_paths = collect_images(args.dir_mask)

#     print(f"[INFO] Found images: {len(image_paths)}")
#     print(f"[INFO] Found masks:  {len(mask_paths)}")

#     if len(image_paths) == 0:
#         raise RuntimeError("No images found. Check --dir_image path.")

#     if len(mask_paths) == 0:
#         raise RuntimeError("No masks found. Check --dir_mask path.")

#     os.makedirs(args.outputs, exist_ok=True)

#     # --- cycle masks if fewer than images ---
#     mask_iter = cycle(mask_paths)

#     # -------------------------
#     # Inference loop
#     # -------------------------
#     for idx, (ipath, mpath) in enumerate(zip(image_paths, mask_iter)):
#         # --- load image ---
#         image = ToTensor()(Image.open(ipath).convert("RGB"))
#         image = image * 2.0 - 1.0
#         image = image.unsqueeze(0).to(device)

#         # --- load mask ---
#         mask = ToTensor()(Image.open(mpath).convert("L"))
#         mask = (mask > 0.5).float()   # 1 = hole
#         mask = mask.unsqueeze(0).to(device)

#         # --- masked input ---
#         image_masked = image * (1.0 - mask)

#         # --- inference ---
#         with torch.no_grad():
#             pred = model(image_masked, mask)

#         # --- final restored image ---
#         restored = image * (1.0 - mask) + pred * mask

#         # --- filenames ---
#         base = os.path.splitext(os.path.basename(ipath))[0]

#         postprocess(image_masked[0]).save(
#             os.path.join(args.outputs, f"{base}_masked.png")
#         )
#         postprocess(pred[0]).save(
#             os.path.join(args.outputs, f"{base}_pred.png")
#         )
#         postprocess(restored[0]).save(
#             os.path.join(args.outputs, f"{base}_restored.png")
#         )

#         if idx % 50 == 0:
#             print(f"[INFO] Processed {idx + 1}/{len(image_paths)}")

#     print("[DONE] Inference completed successfully.")


# if __name__ == "__main__":
#     main_worker(args)


# import importlib
# import os
# from glob import glob
# from itertools import cycle

# import numpy as np
# import torch
# from PIL import Image
# from torchvision.transforms import ToTensor
# from utils.option import args


# def postprocess(image):
#     image = torch.clamp(image, -1.0, 1.0)
#     image = (image + 1) / 2.0 * 255.0
#     image = image.permute(1, 2, 0)
#     image = image.cpu().numpy().astype(np.uint8)
#     return Image.fromarray(image)


# def main_worker(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     net = importlib.import_module("model." + args.model)
#     model = net.InpaintGenerator(args).to(device)
#     model.load_state_dict(torch.load(args.pre_train, map_location=device))
#     model.eval()

#     image_paths = sorted(glob(os.path.join(args.dir_image, "*.jpg")) +
#                          glob(os.path.join(args.dir_image, "*.png")))
#     mask_paths = sorted(glob(os.path.join(args.dir_mask, "*.png")))

#     print(f"Images: {len(image_paths)}, Masks: {len(mask_paths)}")

#     if len(image_paths) == 0 or len(mask_paths) == 0:
#         raise RuntimeError("Images or masks not found")

#     os.makedirs(args.outputs, exist_ok=True)

#     mask_iter = cycle(mask_paths)

#     for ipath, mpath in zip(image_paths, mask_iter):
#         image = ToTensor()(Image.open(ipath).convert("RGB"))
#         image = image * 2.0 - 1.0
#         image = image.unsqueeze(0).to(device)

#         mask = ToTensor()(Image.open(mpath).convert("L"))
#         mask = (mask > 0.5).float()
#         mask = mask.unsqueeze(0).to(device)

#         image_masked = image * (1 - mask)

#         with torch.no_grad():
#             pred = model(image_masked, mask)

#         restored = image * (1 - mask) + pred * mask

#         name = os.path.splitext(os.path.basename(ipath))[0]

#         postprocess(image_masked[0]).save(
#             os.path.join(args.outputs, f"{name}_masked.png")
#         )
#         postprocess(pred[0]).save(
#             os.path.join(args.outputs, f"{name}_pred.png")
#         )
#         postprocess(restored[0]).save(
#             os.path.join(args.outputs, f"{name}_restored.png")
#         )

#         print(f"Saved {name}")


# if __name__ == "__main__":
#     main_worker(args)


























import importlib
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from utils.option import args


def postprocess(image):
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)


def main_worker(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели
    net = importlib.import_module("model." + args.model)
    model = net.InpaintGenerator(args).to(device)
    model.load_state_dict(torch.load(args.pre_train, map_location=device))
    model.eval()

    # Сбор путей
    image_paths = sorted(glob(os.path.join(args.dir_image, "*.jpg")) +
                         glob(os.path.join(args.dir_image, "*.png")))
    mask_paths = sorted(glob(os.path.join(args.dir_mask, "*.png")))

    if len(image_paths) != len(mask_paths):
        print(f"Несоответствие: {len(image_paths)} изображений и {len(mask_paths)} масок")
        return

    os.makedirs(args.outputs, exist_ok=True)

    for ipath, mpath in zip(image_paths, mask_paths):
        # Загрузка изображения
        image = ToTensor()(Image.open(ipath).convert("RGB"))
        image = (image * 2.0 - 1.0).unsqueeze(0).to(device)

        # Загрузка маски (бинарная, 1 = дыра)
        mask = ToTensor()(Image.open(mpath).convert("L"))
        mask = (mask > 0.5).float().unsqueeze(0).to(device)

        # Повреждённое изображение: дыры белого цвета (как ожидает модель)
        image_masked = image * (1 - mask) + mask * 1.0

        with torch.no_grad():
            pred = model(image_masked, mask)

        # Финальное восстановленное (композиция)
        restored = image * (1 - mask) + pred * mask

        name = os.path.splitext(os.path.basename(ipath))[0]

        # Сохраняем только самое важное
        postprocess(restored[0]).save(os.path.join(args.outputs, f"{name}_restored.png"))

        print(f"Обработано и сохранено: {name}")

    print(f"Все результаты в {args.outputs}")


if __name__ == "__main__":
    main_worker(args)