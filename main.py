# ---------------------------
# Standard & Third-Party Imports
# ---------------------------
import itertools
import os
import comet_ml
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.logger import Logger
from ignite.metrics import MaximumMeanDiscrepancy
import copy

# Import the high-performance datamodule from data.py
from data import TransformerCycleGANDataModule, load_salt_model_from_config

# ---------------------------
# TRANSFORMER BACKBONES FOR CYCLEGAN
# ---------------------------

class GeneratorTransformer(nn.Module):
    """
    A Transformer-based generator that modifies ONLY THE FIRST FEATURE (cells_E) of the constituents.
    It takes constituents and a global condition vector, and learns a residual transformation for energy.
    """
    def __init__(self, constituent_features: int, condition_dim: int, embed_dim: int,
                 num_heads: int, num_layers: int, ff_dim: int, max_constituents: int, dropout: float = 0):
        super().__init__()
        self.input_projection = nn.Linear(constituent_features + condition_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True # norm_first is generally more stable
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_projection = nn.Linear(embed_dim, 1)

    def forward(self, constituents: torch.Tensor, mask: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = constituents.shape
        
        condition_expanded = condition.unsqueeze(1).expand(-1, seq_length, -1)
        combined_input = torch.cat([constituents, condition_expanded], dim=-1)
        
        x = self.input_projection(combined_input)
        
        transformed_x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        delta_E = self.output_projection(transformed_x)
        delta_constituents = torch.zeros_like(constituents)
        delta_constituents[:, :, 0] = delta_E.squeeze(-1)

        return constituents + delta_constituents


class DiscriminatorTransformer(nn.Module):
    """
    A Transformer-based discriminator (critic) for distinguishing real vs. fake sets of jet constituents.
    """
    def __init__(self, constituent_features: int, condition_dim: int, embed_dim: int,
                 num_heads: int, num_layers: int, ff_dim: int, max_constituents: int, dropout: float = 0):
        super().__init__()
        self.input_projection = nn.Linear(constituent_features + condition_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True # norm_first is generally more stable
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.critic_head = nn.Sequential(
            # nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, constituents: torch.Tensor, mask: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = constituents.shape
        condition_expanded = condition.unsqueeze(1).expand(-1, seq_length, -1)
        combined_input = torch.cat([constituents, condition_expanded], dim=-1)

        x = self.input_projection(combined_input)
        
        transformed_x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Masked average pooling to get a single vector per jet
        mask_expanded = ~mask.unsqueeze(-1) # Invert mask (True for valid tokens) and expand
        pooled_input = transformed_x * mask_expanded
        num_valid_tokens = mask_expanded.sum(dim=1)
        pooled_output = pooled_input.sum(dim=1) / (num_valid_tokens + 1e-8)
        
        score = self.critic_head(pooled_output)
        return score

# ---------------------------
# MAIN LIGHTNING MODULE
# ---------------------------

class TransformerCycleGANLightning(pl.LightningModule):
    def __init__(
        self,
        lr_config: dict = {"initial_lr": 1e-4, "scheduler_class": None, "scheduler_params": {}},
        optimizer_config: dict = {"optimizer_class": "AdamW", "optimizer_params": {}},
        cycleGAN_config: dict = {
            "constituent_features": None, "condition_dim": None, "max_constituents": None,
            "embed_dim": 128, "num_heads": 4, "num_layers": 4, "ff_dim": 512,
            "lambda_a": 10.0, "lambda_b": 10.0, "lambda_id": 0.5,
            "lambda_energy": 0.0, "lambda_gp": 10.0,
        },
        salt_config: dict = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        cfg = self.hparams.cycleGAN_config
        self.lambda_a = cfg['lambda_a']
        self.lambda_b = cfg['lambda_b']
        self.lambda_id = cfg['lambda_id']
        self.lambda_gp = cfg['lambda_gp']
        self.lambda_energy = cfg['lambda_energy']

        transformer_args = {
            "constituent_features": cfg['constituent_features'],
            "condition_dim": cfg['condition_dim'], "embed_dim": cfg['embed_dim'],
            "num_heads": cfg['num_heads'], "num_layers": cfg['num_layers'],
            "ff_dim": cfg['ff_dim'], "max_constituents": cfg['max_constituents']
        }

        self.netG_A = GeneratorTransformer(**transformer_args)
        self.netG_B = GeneratorTransformer(**transformer_args)
        self.netD_A = DiscriminatorTransformer(**transformer_args)
        self.netD_B = DiscriminatorTransformer(**transformer_args)

        self.criterionCycle = nn.L1Loss()
        self.criterionIdentity = nn.L1Loss()
        
        self.register_buffer('const_mean', torch.zeros(cfg['constituent_features']))
        self.register_buffer('const_std', torch.ones(cfg['constituent_features']))
        self.register_buffer('cond_mean', torch.zeros(cfg['condition_dim']))
        self.register_buffer('cond_std', torch.ones(cfg['condition_dim']))

    def setup(self, stage: str):
        if stage == 'fit':
            print("Fetching shared normalization statistics from datamodule.")
            dm = self.trainer.datamodule
            self.const_mean = dm.const_mean.clone().to(self.device)
            self.const_std = dm.const_std.clone().to(self.device)
            self.cond_mean = dm.cond_mean.clone().to(self.device)
            self.cond_std = dm.cond_std.clone().to(self.device)

        if self.hparams.salt_config and self.lambda_energy > 0 and not hasattr(self, 'energy_MLP'):
            energy_model, _ = load_salt_model_from_config(
                self.hparams.salt_config["base_config"],
                self.hparams.salt_config["user_config"],
                self.hparams.salt_config["ckpt_path"]
            )
            self.energy_MLP = copy.deepcopy(energy_model.model.tasks[0].net)
            self.set_requires_grad(self.energy_MLP, False)
            self.energy_loss_fn = nn.L1Loss()
            del energy_model

    def _normalize(self, tensor, mean, std):
        return (tensor - mean) / (std + 1e-8)

    def _denormalize(self, tensor, mean, std):
        return tensor * (std + 1e-8) + mean
    
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list): nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def _masked_global_avg_pool(self, constituents, mask):
        mask_expanded = ~mask.unsqueeze(-1)
        pooled_input = constituents * mask_expanded
        num_valid_tokens = mask_expanded.sum(dim=1)
        return pooled_input.sum(dim=1) / (num_valid_tokens + 1e-8)

    def forward(self, consts_A, mask_A, consts_B, mask_B, cond_A, cond_B):
        consts_A_norm = self._normalize(consts_A, self.const_mean, self.const_std)
        consts_B_norm = self._normalize(consts_B, self.const_mean, self.const_std)
        cond_A_norm = self._normalize(cond_A, self.cond_mean, self.cond_std)
        cond_B_norm = self._normalize(cond_B, self.cond_mean, self.cond_std)
        
        fake_B_norm = self.netG_A(consts_A_norm, mask_A, cond_A_norm)
        rec_A_norm = self.netG_B(fake_B_norm, mask_A, cond_A_norm)
        fake_A_norm = self.netG_B(consts_B_norm, mask_B, cond_B_norm)
        rec_B_norm = self.netG_A(fake_A_norm, mask_B, cond_B_norm)
        
        fake_B = self._denormalize(fake_B_norm, self.const_mean, self.const_std)
        fake_A = self._denormalize(fake_A_norm, self.const_mean, self.const_std)
        rec_A = self._denormalize(rec_A_norm, self.const_mean, self.const_std)
        rec_B = self._denormalize(rec_B_norm, self.const_mean, self.const_std)
        return fake_A, fake_B, rec_A, rec_B

    def compute_energy_loss(self, real_samples, fake_samples, real_mask, fake_mask):
        if hasattr(self, 'energy_MLP') and self.lambda_energy > 0:
            pooled_real = self._masked_global_avg_pool(real_samples, real_mask)
            pooled_fake = self._masked_global_avg_pool(fake_samples, fake_mask)
            with torch.no_grad():
                energy_real = self.energy_MLP(pooled_real)
            energy_fake = self.energy_MLP(pooled_fake)
            return self.energy_loss_fn(energy_real, energy_fake) * self.lambda_energy
        return 0.0

    def compute_gradient_penalty(self, netD, real_samples, fake_samples, mask, cond):
        alpha = torch.rand((real_samples.size(0), 1, 1), device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = netD(interpolates, mask, cond)
        fake_output = torch.ones(d_interpolates.size(), device=self.device, requires_grad=False)
        gradients = torch.autograd.grad(
            outputs=d_interpolates, inputs=interpolates, grad_outputs=fake_output,
            create_graph=True, retain_graph=True, only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        grad_norm = gradients.norm(2, dim=1)
        return ((grad_norm - 1) ** 2).mean()

    def training_step(self, batch, batch_idx):
        (jets_A, jets_B, real_A, real_B, mask_A, mask_B) = batch
        
        opt_G, opt_D = self.optimizers()
        
        real_A_norm = self._normalize(real_A, self.const_mean, self.const_std)
        real_B_norm = self._normalize(real_B, self.const_mean, self.const_std)
        cond_A_norm = self._normalize(jets_A, self.cond_mean, self.cond_std)
        cond_B_norm = self._normalize(jets_B, self.cond_mean, self.cond_std)
        
        fake_B_norm = self.netG_A(real_A_norm, mask_A, cond_A_norm)
        fake_A_norm = self.netG_B(real_B_norm, mask_B, cond_B_norm)

        # === Train Discriminators ===
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        opt_D.zero_grad()
        
        loss_D_A = self.netD_A(fake_B_norm.detach(), mask_A, cond_A_norm).mean() - self.netD_A(real_B_norm, mask_B, cond_B_norm).mean()
        loss_D_B = self.netD_B(fake_A_norm.detach(), mask_B, cond_B_norm).mean() - self.netD_B(real_A_norm, mask_A, cond_A_norm).mean()
        
        loss_D_total = loss_D_A + loss_D_B
        self.manual_backward(loss_D_total)
        opt_D.step()
        self.log_dict({"train/D_A": loss_D_A, "train/D_B": loss_D_B, "train/D_total": loss_D_total})

        # === Train Generators ===
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        opt_G.zero_grad()

        loss_G_A = -self.netD_A(fake_B_norm, mask_A, cond_A_norm).mean()
        loss_G_B = -self.netD_B(fake_A_norm, mask_B, cond_B_norm).mean()

        rec_A_norm = self.netG_B(fake_B_norm, mask_A, cond_A_norm)
        loss_cycle_A = self.criterionCycle(rec_A_norm[:, :, 0], real_A_norm[:, :, 0]) * self.lambda_a
        rec_B_norm = self.netG_A(fake_A_norm, mask_B, cond_B_norm)
        loss_cycle_B = self.criterionCycle(rec_B_norm[:, :, 0], real_B_norm[:, :, 0]) * self.lambda_b

        id_output_A_norm = self.netG_A(real_B_norm, mask_B, cond_B_norm)
        loss_id_A = self.criterionIdentity(id_output_A_norm[:, :, 0], real_B_norm[:, :, 0]) * self.lambda_id * self.lambda_b
        id_output_B_norm = self.netG_B(real_A_norm, mask_A, cond_A_norm)
        loss_id_B = self.criterionIdentity(id_output_B_norm[:, :, 0], real_A_norm[:, :, 0]) * self.lambda_id * self.lambda_a

        fake_B_denorm = self._denormalize(fake_B_norm, self.const_mean, self.const_std)
        energy_loss_A = self.compute_energy_loss(real_A, fake_B_denorm, mask_A, mask_A)
        fake_A_denorm = self._denormalize(fake_A_norm, self.const_mean, self.const_std)
        energy_loss_B = self.compute_energy_loss(real_B, fake_A_denorm, mask_B, mask_B)

        loss_G_total = (loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B +
                        loss_id_A + loss_id_B + energy_loss_A + energy_loss_B)

        self.manual_backward(loss_G_total)
        torch.nn.utils.clip_grad_norm_(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), 1.0)
        opt_G.step()
        
        self.log_dict({
            "train/G_A_adv": loss_G_A, "train/G_B_adv": loss_G_B, "train/cycle_A": loss_cycle_A,
            "train/cycle_B": loss_cycle_B, "train/id_A": loss_id_A, "train/id_B": loss_id_B,
            "train/energy_A": energy_loss_A, "train/energy_B": energy_loss_B,
            "train/G_total": loss_G_total
        }, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        (jets_A, jets_B, real_A, real_B, mask_A, mask_B) = batch
        
        real_A_norm = self._normalize(real_A, self.const_mean, self.const_std)
        real_B_norm = self._normalize(real_B, self.const_mean, self.const_std)
        cond_A_norm = self._normalize(jets_A, self.cond_mean, self.cond_std)
        cond_B_norm = self._normalize(jets_B, self.cond_mean, self.cond_std)

        fake_B_norm = self.netG_A(real_A_norm, mask_A, cond_A_norm)
        rec_A_norm = self.netG_B(fake_B_norm, mask_A, cond_A_norm)
        fake_A_norm = self.netG_B(real_B_norm, mask_B, cond_B_norm)
        rec_B_norm = self.netG_A(fake_A_norm, mask_B, cond_B_norm)
        
        loss_G_A = -self.netD_A(fake_B_norm, mask_A, cond_A_norm).mean()
        loss_G_B = -self.netD_B(fake_A_norm, mask_B, cond_B_norm).mean()
        
        loss_cycle_A = self.criterionCycle(rec_A_norm[:, :, 0], real_A_norm[:, :, 0]) * self.lambda_a
        loss_cycle_B = self.criterionCycle(rec_B_norm[:, :, 0], real_B_norm[:, :, 0]) * self.lambda_b

        loss_G_total = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B
        self.log_dict({
            "val/G_total": loss_G_total, "val/G_A_adv": loss_G_A, "val/G_B_adv": loss_G_B,
            "val/cycle_A": loss_cycle_A, "val/cycle_B": loss_cycle_B,
        }, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        lr_cfg = self.hparams.lr_config
        opt_cfg = self.hparams.optimizer_config
        optimizer_class = getattr(torch.optim, opt_cfg['optimizer_class'])
        if 'betas' not in opt_cfg['optimizer_params']:
             opt_cfg['optimizer_params']['betas'] = (0.5, 0.999)
        opt_G = optimizer_class(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=lr_cfg['initial_lr'], **opt_cfg['optimizer_params'])
        opt_D = optimizer_class(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=lr_cfg['initial_lr'], **opt_cfg['optimizer_params'])
        optimizers = [opt_G, opt_D]
        if lr_cfg.get('scheduler_class') is not None:
            scheduler_class = getattr(torch.optim.lr_scheduler, lr_cfg['scheduler_class'])
            sched_G = scheduler_class(opt_G, **lr_cfg['scheduler_params'])
            sched_D = scheduler_class(opt_D, **lr_cfg['scheduler_params'])
            return optimizers, [sched_G, sched_D]
        return optimizers

# ---------------------------
# CLI HELPER CLASS
# ---------------------------
class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, Logger) and trainer.logger.save_dir is not None:
            save_dir = trainer.logger.save_dir
            name = trainer.logger.name
            version = trainer.logger.version
            version = version if isinstance(version, str) else f"version_{version}"
            config_path = os.path.join(save_dir, str(name), version, "config.yaml")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            self.parser.save(
                self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
            )
            print(f"Config saved to: {config_path}")

# ---------------------------
# CLI ENTRYPOINT
# ---------------------------
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision('highest')
    cli = LightningCLI(
        model_class=TransformerCycleGANLightning,
        datamodule_class=TransformerCycleGANDataModule,
        run=True,
        save_config_callback=LoggerSaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        auto_configure_optimizers=False,
    )