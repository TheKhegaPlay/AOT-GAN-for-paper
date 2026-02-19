from torch.utils.data import DataLoader
import torch

from .dataset import InpaintingData


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def create_loader(args):
    dataset = InpaintingData(args)
    
    # ===== КРИТИЧНО: правильные настройки для Windows =====
    # На Windows num_workers > 0 может медленнить (multiprocessing overhead)
    # pin_memory = True ускоряет CPU→GPU трансфер если CUDA доступна
    # num_workers = 0 if args.distributed else 4  # на Windows лучше 0 или 2
    # pin_memory = torch.cuda.is_available()
    
    # data_loader = DataLoader(
    #     dataset,
    #     batch_size=args.batch_size // args.world_size,
    #     shuffle=True,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    #     persistent_workers=(num_workers > 0),  # только если num_workers > 0
    #     drop_last=True,  # ===== НОВОЕ: отбросить неполный батч =====
    #     prefetch_factor=2 if num_workers > 0 else None,  # только если multiprocessing
    # )
    num_workers = 2
    pin_memory = True

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size // args.world_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        prefetch_factor=2,
    )


    return sample_data(data_loader)
