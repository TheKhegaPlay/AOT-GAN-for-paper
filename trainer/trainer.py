import importlib
import os
from glob import glob
import signal
import traceback

import torch
from data import create_loader
from loss import loss as loss_module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from .common import timer


class Trainer:
    def __init__(self, args):
        self.args = args
        self.iteration = 0
        
        # ===== КРИТИЧНО: явный выбор device =====
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Training on device: {self.device}")
        if torch.cuda.is_available():
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] CUDA Version: {torch.version.cuda}")

        # setup data set and data loader
        self.dataloader = create_loader(args)

        # set up losses and metrics
        self.rec_loss_func = {key: getattr(loss_module, key)() for key, val in args.rec_loss.items()}
        self.adv_loss = getattr(loss_module, args.gan_type)()
        
        # ========== НОВОЕ: Confidence-aware losses для HFI-Gen ==========
        # Инициализируем confidence losses если они есть в loss module
        try:
            self.confidence_reg_loss = getattr(loss_module, 'ConfidenceRegularization')(device=self.device)
            self.weighted_recon_loss = getattr(loss_module, 'WeightedReconstructionLoss')(device=self.device)
            print("[INFO] Confidence-aware losses initialized (HFI-Gen mode)")
        except AttributeError:
            # Confidence losses не доступны, работаем в совместимом режиме
            self.confidence_reg_loss = None
            self.weighted_recon_loss = None
            print("[INFO] Confidence-aware losses not found (baseline mode)")

        # Image generator input: [rgb(3) + mask(1)], discriminator input: [rgb(3)]
        net = importlib.import_module("model." + args.model)

        # ===== ИСПРАВЛЕНО: .to(device) вместо .cuda() =====
        self.netG = net.InpaintGenerator(args).to(self.device)
        self.optimG = torch.optim.Adam(self.netG.parameters(), lr=args.lrg, betas=(args.beta1, args.beta2))

        self.netD = net.Discriminator().to(self.device)
        self.optimD = torch.optim.Adam(self.netD.parameters(), lr=args.lrd, betas=(args.beta1, args.beta2))

        self.load()
        if args.distributed:
            self.netG = DDP(self.netG, device_ids=[args.local_rank], output_device=[args.local_rank])
            self.netD = DDP(self.netD, device_ids=[args.local_rank], output_device=[args.local_rank])

        if args.tensorboard:
            self.writer = SummaryWriter(os.path.join(args.save_dir, "log"))
        
        # ===== НОВОЕ: graceful shutdown handler =====
        self._setup_signal_handlers()

        # ===== НОВОЕ: детальный лог GPU =====
        if self.args.global_rank == 0:
            print(f"\n{'='*60}")
            print(f"[SETUP] Training Configuration")
            print(f"{'='*60}")
            print(f"Device: {self.device}")
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"GPU: {gpu_name}")
                print(f"GPU Memory: {gpu_mem:.1f} GB")
                print(f"CUDA Version: {torch.version.cuda}")
            else:
                print("WARNING: CUDA not available, training on CPU (very slow)")
            print(f"Batch Size: {args.batch_size}")
            print(f"Learning Rate (G): {args.lrg}")
            print(f"Learning Rate (D): {args.lrd}")
            print(f"Save Every: {int(args.save_every)} iterations")
            print(f"Total Iterations: {int(args.iterations)}")
            print(f"{'='*60}\n")

    def _setup_signal_handlers(self):
        """Сохранить модель при Ctrl+C"""
        def signal_handler(sig, frame):
            print("\n[!] KeyboardInterrupt detected. Saving checkpoint...")
            self.save()
            print("[✓] Checkpoint saved. Exiting.")
            exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)

    def load(self):
        try:
            gpath = sorted(glob(os.path.join(self.args.save_dir, "G*.pt")))[-1]
            self.netG.load_state_dict(torch.load(gpath, map_location="cuda"))
            self.iteration = int(os.path.basename(gpath)[1:-3])
            if self.args.global_rank == 0:
                print(f"[**] Loading generator network from {gpath}")
        except Exception:
            pass

        try:
            dpath = sorted(glob(os.path.join(self.args.save_dir, "D*.pt")))[-1]
            self.netD.load_state_dict(torch.load(dpath, map_location="cuda"))
            if self.args.global_rank == 0:
                print(f"[**] Loading discriminator network from {dpath}")
        except Exception:
            pass

        try:
            opath = sorted(glob(os.path.join(self.args.save_dir, "O*.pt")))[-1]
            data = torch.load(opath, map_location="cuda")
            self.optimG.load_state_dict(data["optimG"])
            self.optimD.load_state_dict(data["optimD"])
            if self.args.global_rank == 0:
                print(f"[**] Loading optimizer from {opath}")
        except Exception:
            pass

    def save(
        self,
    ):
        if self.args.global_rank == 0:
            # ===== НОВОЕ: явное сохранение с проверкой =====
            try:
                print(f"\n[SAVE] iteration {self.iteration}...", end=" ", flush=True)
                
                # сохранить на CPU memory перво, потом на диск (более надёжно на Windows)
                g_state = self.netG.module.state_dict() if isinstance(self.netG, DDP) else self.netG.state_dict()
                d_state = self.netD.module.state_dict() if isinstance(self.netD, DDP) else self.netD.state_dict()
                opt_state = {
                    "optimG": self.optimG.state_dict(),
                    "optimD": self.optimD.state_dict(),
                    "iteration": self.iteration,
                }
                
                g_path = os.path.join(self.args.save_dir, f"G{str(self.iteration).zfill(7)}.pt")
                d_path = os.path.join(self.args.save_dir, f"D{str(self.iteration).zfill(7)}.pt")
                o_path = os.path.join(self.args.save_dir, f"O{str(self.iteration).zfill(7)}.pt")
                
                torch.save(g_state, g_path)
                torch.save(d_state, d_path)
                torch.save(opt_state, o_path)
                
                print("✓")
            except Exception as e:
                print(f"✗ FAILED: {e}")
                raise

    def train(self):
        pbar = range(self.iteration, self.args.iterations)
        if self.args.global_rank == 0:
            pbar = tqdm(range(self.args.iterations), initial=self.iteration, dynamic_ncols=True, smoothing=0.01)
            timer_data, timer_model = timer(), timer()

        try:
            for idx in pbar:
                self.iteration += 1
                images, masks, filename = next(self.dataloader)
                images, masks = images.to(self.device), masks.to(self.device)
                images_masked = (images * (1 - masks).float()) + masks

                if self.args.global_rank == 0:
                    timer_data.hold()
                    timer_model.tic()

                # ========== FORWARD PASS WITH HYBRID GENERATOR ==========
                gen_output = self.netG(images_masked, masks)
                
                # Поддержка ОБЕИХ архитектур: с confidence branch и без
                if isinstance(gen_output, tuple):
                    # Новый HFI-Gen с confidence branch
                    pred_img, confidence_map = gen_output
                    has_confidence = True
                else:
                    # Оригинальный AOT-GAN без confidence branch
                    pred_img = gen_output
                    confidence_map = None
                    has_confidence = False
                
                comp_img = (1 - masks) * images + masks * pred_img

                # ========== RECONSTRUCTION LOSSES ==========
                losses = {}
                for name, weight in self.args.rec_loss.items():
                    losses[name] = weight * self.rec_loss_func[name](pred_img, images)

                # ========== CONFIDENCE-AWARE LOSSES (если модель их поддерживает) ==========
                if has_confidence and self.confidence_reg_loss is not None:
                    # Confidence regularization: штрафуем модель за неправильную уверенность
                    conf_reg_loss = self.confidence_reg_loss(confidence_map, masks)
                    losses["conf_reg"] = conf_reg_loss * getattr(self.args, 'conf_reg_weight', 0.3)
                    
                    # Weighted reconstruction loss: взвешиваем по confidence
                    if self.weighted_recon_loss is not None:
                        w_recon_loss = self.weighted_recon_loss(pred_img, images, confidence_map, masks)
                        losses["w_recon"] = w_recon_loss * getattr(self.args, 'w_recon_weight', 0.2)

                # ========== ADVERSARIAL LOSS ==========
                dis_loss, gen_loss = self.adv_loss(self.netD, comp_img, images, masks)
                losses["advg"] = gen_loss * self.args.adv_weight

                # ========== BACKPROPAGATION ==========
                self.optimG.zero_grad()
                self.optimD.zero_grad()
                sum(losses.values()).backward()
                losses["advd"] = dis_loss
                dis_loss.backward()
                self.optimG.step()
                self.optimD.step()

                if self.args.global_rank == 0:
                    timer_model.hold()
                    timer_data.tic()

                # ========== LOGGING & VISUALIZATION ==========
                if self.args.global_rank == 0 and (self.iteration % self.args.print_every == 0):
                    pbar.update(self.args.print_every)
                    description = f"mt:{timer_model.release():.1f}s, dt:{timer_data.release():.1f}s, "
                    for key, val in losses.items():
                        description += f"{key}:{val.item():.3f}, "
                        if self.args.tensorboard:
                            self.writer.add_scalar(key, val.item(), self.iteration)
                    pbar.set_description(description)
                    
                    if self.args.tensorboard:
                        self.writer.add_image("mask", make_grid(masks), self.iteration)
                        self.writer.add_image("orig", make_grid((images + 1.0) / 2.0), self.iteration)
                        self.writer.add_image("pred", make_grid((pred_img + 1.0) / 2.0), self.iteration)
                        self.writer.add_image("comp", make_grid((comp_img + 1.0) / 2.0), self.iteration)
                        
                        # Добавить визуализацию confidence map если доступна
                        if has_confidence and confidence_map is not None:
                            self.writer.add_image("confidence", make_grid(confidence_map), self.iteration)

                # ========== AUTOSAVE ==========
                if self.args.global_rank == 0 and (self.iteration % self.args.save_every) == 0:
                    self.save()
        
        except KeyboardInterrupt:
            if self.args.global_rank == 0:
                print("\n[!] Training interrupted by user. Saving checkpoint...")
                self.save()
                print(f"[✓] Checkpoint saved at iteration {self.iteration}")
            raise
        except Exception as e:
            if self.args.global_rank == 0:
                print(f"\n[ERROR] Exception during training: {e}")
                print("[!] Attempting emergency save...")
                traceback.print_exc()
                try:
                    self.save()
                    print("[✓] Emergency checkpoint saved")
                except Exception as save_err:
                    print(f"[✗] Emergency save failed: {save_err}")
            raise
