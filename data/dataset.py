import os
from glob import glob

import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


class InpaintingData(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.w = self.h = args.image_size
        self.mask_type = args.mask_type

        # image and mask
        self.image_path = []
        
        # Попробуем разные варианты пути к изображениям
        image_dir_candidates = [
            os.path.join(args.dir_image, args.data_train),  # исходный вариант
            args.dir_image,  # прямой путь к папке
        ]
        
        for candidate_dir in image_dir_candidates:
            if os.path.isdir(candidate_dir):
                for ext in ["*.jpg", "*.png"]:
                    self.image_path.extend(glob(os.path.join(candidate_dir, ext)))
            if self.image_path:
                break
        
        # Попытаемся найти маски
        mask_dir_candidates = [
            os.path.join(args.dir_mask, args.mask_type),  # исходный вариант
            args.dir_mask,  # прямой путь
        ]
        
        self.mask_path = []
        for candidate_dir in mask_dir_candidates:
            if os.path.isdir(candidate_dir):
                self.mask_path = glob(os.path.join(candidate_dir, "*.png"))
            if self.mask_path:
                break

        if not self.image_path:
            print(f"[WARNING] No images found in {args.dir_image}")
        if not self.mask_path:
            print(f"[WARNING] No masks found in {args.dir_mask}")

        # augmentation
        self.img_trans = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.ToTensor(),
            ]
        )
        self.mask_trans = transforms.Compose(
            [
                transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((0, 45), interpolation=transforms.InterpolationMode.NEAREST),
            ]
        )

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        # load image
        image = Image.open(self.image_path[index]).convert("RGB")
        filename = os.path.basename(self.image_path[index])

        if self.mask_type == "pconv" and self.mask_path:
            index = np.random.randint(0, len(self.mask_path))
            mask = Image.open(self.mask_path[index])
            mask = mask.convert("L")
        else:
            mask = np.zeros((self.h, self.w)).astype(np.uint8)
            mask[self.h // 4 : self.h // 4 * 3, self.w // 4 : self.w // 4 * 3] = 1
            mask = Image.fromarray(mask).convert("L")

        # augment
        image = self.img_trans(image) * 2.0 - 1.0
        mask = F.to_tensor(self.mask_trans(mask))

        return image, mask, filename


if __name__ == "__main__":
    from attrdict import AttrDict

    args = {
        "dir_image": "../../../dataset",
        "data_train": "places2",
        "dir_mask": "../../../dataset",
        "mask_type": "pconv",
        "image_size": 512,
    }
    args = AttrDict(args)

    data = InpaintingData(args)
    print(len(data), len(data.mask_path))
    img, mask, filename = data[0]
    print(img.size(), mask.size(), filename)
