"""
training/trainer.py
Main training loop for the geometry-aware face swapping system.

Orchestrates:
  - Multi-dataset dataloading
  - Generator (U-Net + ControlNet + region encoders) training
  - Discriminator (PatchGAN) training
  - Mixed precision via HuggingFace Accelerate
  - EMA weight averaging
  - Gradient accumulation
  - Checkpoint saving / resuming
  - Validation loop with metric logging
  - TensorBoard / W&B logging (via Accelerate)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
except ImportError:
    Accelerator = None  # type: ignore[assignment, misc]

try:
    from diffusers.training_utils import EMAModel
except ImportError:
    EMAModel = None  # type: ignore[assignment, misc]

from data.dataset import build_dataloader
from losses.geometry_loss import GeometryConsistencyLoss, PixelReconstructionLoss
from losses.identity_loss import IdentityLoss
from losses.perceptual_loss import PerceptualLoss
from models.backbone import FaceSwapBackbone
from models.controlnet import GeometryControlNet
from models.discriminator import MultiScaleDiscriminator, hinge_d_loss, hinge_g_loss
from models.geometry import GeometryConditioning
from models.region_encoder import FaceRegionEncoder
from training.scheduler import (
    build_discriminator_optimiser,
    build_generator_optimiser,
    build_scheduler,
)
from utils.face_crop import FaceRegionCropper
from utils.metrics import compute_metrics
from utils.visualize import save_comparison_grid


class FaceSwapTrainer:
    """
    Full training harness for geometry-aware face swapping.

    Args:
        config_path (str | Path): Path to train_config.yaml.
        output_dir (Optional[str]): Override output directory.
    """

    def __init__(
        self,
        config_path: str | Path,
        output_dir: Optional[str] = None,
    ) -> None:
        # ── Load config ──────────────────────────────────────────────────────
        with open(config_path) as f:
            self.cfg: Dict[str, Any] = yaml.safe_load(f)

        if output_dir:
            self.cfg["experiment"]["output_dir"] = output_dir
        self.output_dir = Path(self.cfg["experiment"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Accelerator ──────────────────────────────────────────────────────
        if Accelerator is None:
            raise ImportError("accelerate is required. pip install accelerate")

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg["training"]["gradient_accumulation_steps"],
            mixed_precision=self.cfg["training"]["mixed_precision"],
            log_with="tensorboard",
            project_dir=str(self.output_dir),
        )
        set_seed(self.cfg["experiment"]["seed"])

        self.device = self.accelerator.device
        dtype_str = self.cfg["training"]["mixed_precision"]
        self.dtype = torch.float16 if dtype_str == "fp16" else (
            torch.bfloat16 if dtype_str == "bf16" else torch.float32
        )

        self._build_models()
        self._build_losses()
        self._build_dataloaders()
        self._build_optimisers()
        self._wrap_with_accelerator()

        # EMA
        ema_cfg = self.cfg["training"].get("ema", {})
        if ema_cfg.get("enabled", False) and EMAModel is not None:
            self.ema_unet = EMAModel(
                parameters=self.backbone.unet.parameters(),
                decay=ema_cfg.get("decay", 0.9999),
            )
        else:
            self.ema_unet = None

        self.global_step: int = 0
        self._resume_checkpoint()

    # ── Model construction ───────────────────────────────────────────────────

    def _build_models(self) -> None:
        """Instantiate all model components."""
        mcfg = self.cfg["model"]
        tcfg = self.cfg["training"]

        self.backbone = FaceSwapBackbone(
            sdxl_model_id=mcfg["sdxl_model_id"],
            vae_model_id=mcfg.get("vae_model_id"),
            region_projection_dim=mcfg["region_encoder"]["projection_dim"],
            regions=mcfg["region_encoder"]["regions"],
            attn_scale=mcfg["cross_attention"]["scale"],
            freeze_unet=False,
            use_xformers=True,
            device=str(self.device),
            dtype=self.dtype,
        )

        rec_cfg = mcfg["region_encoder"]
        self.region_encoder = FaceRegionEncoder(
            vit_model_name=rec_cfg["vit_model"],
            projection_dim=rec_cfg["projection_dim"],
            num_tokens=mcfg["cross_attention"]["num_tokens"],
            regions=rec_cfg["regions"],
        )

        geo_cfg = mcfg["controlnet"]
        self.controlnet = GeometryControlNet(
            conditioning_channels=geo_cfg["conditioning_channels"],
            block_out_channels=tuple(geo_cfg["block_out_channels"]),
        )

        self.geometry = GeometryConditioning(
            deca_model_path=mcfg["deca_model_path"],
            deca_cfg_path=mcfg["deca_cfg_path"],
            device=str(self.device),
        )

        disc_cfg = mcfg["discriminator"]
        self.discriminator = MultiScaleDiscriminator(
            num_scales=3,
            ndf=disc_cfg["ndf"],
            n_layers=disc_cfg["n_layers"],
            use_spectral_norm=disc_cfg["use_spectral_norm"],
        )

        self.cropper = FaceRegionCropper()

        if tcfg.get("gradient_checkpointing", False):
            self.backbone.unet.enable_gradient_checkpointing()

    def _build_losses(self) -> None:
        """Instantiate loss functions."""
        lcfg = self.cfg["losses"]
        mcfg = self.cfg["model"]

        self.identity_loss = IdentityLoss(
            model_path=mcfg["arcface_model_path"],
            weight=lcfg["identity"]["weight"],
        )
        self.perceptual_loss = PerceptualLoss(
            weight=lcfg["perceptual"]["weight"],
            layer_names=lcfg["perceptual"]["layers"],
        )
        self.pixel_loss = PixelReconstructionLoss(
            weight=lcfg["pixel_reconstruction"]["weight"],
        )
        self.geometry_loss = GeometryConsistencyLoss(
            deca_model_path=mcfg["deca_model_path"],
            deca_cfg_path=mcfg["deca_cfg_path"],
            weight=lcfg["geometry_consistency"]["weight"],
            device=str(self.device),
        )
        self.adv_weight: float = lcfg["adversarial"]["weight"]

    def _build_dataloaders(self) -> None:
        self.train_loader = build_dataloader(self.cfg, split="train", cropper=self.cropper)
        self.val_loader = build_dataloader(self.cfg, split="val", cropper=self.cropper)

    def _build_optimisers(self) -> None:
        scfg = self.cfg["training"]["scheduler"]
        total_steps = self.cfg["training"]["total_steps"]
        warmup = scfg["warmup_steps"]

        self.gen_opt = build_generator_optimiser(
            self.backbone, self.region_encoder, self.controlnet, self.cfg
        )
        self.disc_opt = build_discriminator_optimiser(self.discriminator, self.cfg)

        self.gen_scheduler = build_scheduler(
            self.gen_opt,
            scheduler_type=scfg["type"],
            warmup_steps=warmup,
            total_steps=total_steps,
            num_cycles=scfg.get("num_cycles", 1),
        )
        self.disc_scheduler = build_scheduler(
            self.disc_opt,
            scheduler_type=scfg["type"],
            warmup_steps=warmup,
            total_steps=total_steps,
        )

    def _wrap_with_accelerator(self) -> None:
        """Prepare models, optimisers, and dataloaders with Accelerate."""
        (
            self.backbone.unet,
            self.region_encoder,
            self.controlnet,
            self.discriminator,
            self.gen_opt,
            self.disc_opt,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(
            self.backbone.unet,
            self.region_encoder,
            self.controlnet,
            self.discriminator,
            self.gen_opt,
            self.disc_opt,
            self.train_loader,
            self.val_loader,
        )

    # ── Checkpoint helpers ───────────────────────────────────────────────────

    def _resume_checkpoint(self) -> None:
        if resume_path := self.cfg["experiment"].get("resume_from_checkpoint"):
            self.accelerator.print(f"Resuming from: {resume_path}")
            self.accelerator.load_state(resume_path)
            self.global_step = int(Path(resume_path).name.split("-")[-1])

    def _save_checkpoint(self) -> None:
        ckpt_dir = self.output_dir / f"checkpoint-{self.global_step}"
        self.accelerator.save_state(str(ckpt_dir))
        if self.ema_unet is not None:
            self.ema_unet.save_pretrained(str(ckpt_dir / "unet_ema"))
        self._prune_old_checkpoints()

    def _prune_old_checkpoints(self) -> None:
        """Keep only the last N checkpoints."""
        n = self.cfg["training"].get("keep_last_n_checkpoints", 3)
        ckpts = sorted(self.output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        for old in ckpts[:-n]:
            import shutil
            shutil.rmtree(old, ignore_errors=True)

    # ── Single training step ─────────────────────────────────────────────────

    def _training_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Execute one generator + discriminator update step.

        Args:
            batch: Collated batch dict from dataloader.

        Returns:
            Dict of scalar loss tensors for logging.
        """
        src_images = batch["source_image"].to(self.device, dtype=self.dtype)
        tgt_images = batch["target_image"].to(self.device, dtype=self.dtype)
        region_crops = {k: v.to(self.device, dtype=self.dtype)
                        for k, v in batch["source_regions"].items()}

        # ── Encode to latent space ────────────────────────────────────────
        with torch.no_grad():
            src_latents = self.backbone.encode_images(src_images)
            tgt_latents = self.backbone.encode_images(tgt_images)

        # ── Geometry conditioning ─────────────────────────────────────────
        with self.accelerator.autocast():
            geo_out = self.geometry(F.interpolate(src_images, (224, 224)))
            depth_map = geo_out["depth_map"]
            param_emb = geo_out["param_embedding"]
            controlnet_out = self.controlnet(depth_map, param_emb)

        # ── Region features ───────────────────────────────────────────────
        with self.accelerator.autocast():
            region_feats = self.region_encoder(region_crops)

        # ── Diffusion noise ───────────────────────────────────────────────
        noise = torch.randn_like(tgt_latents)
        B = src_latents.shape[0]
        timesteps = torch.randint(
            0,
            self.backbone.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=self.device,
        ).long()
        noisy_latents = self.backbone.add_noise(tgt_latents, noise, timesteps)

        # ── Text conditioning (empty prompts during pure identity training)
        with torch.no_grad():
            prompt_embeds, pooled_embeds = self.backbone.encode_text([""] * B)

        time_ids = torch.zeros(B, 6, device=self.device)
        added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": time_ids}

        # ── Generator forward ─────────────────────────────────────────────
        with self.accelerator.autocast():
            noise_pred = self.backbone(
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                controlnet_output=controlnet_out,
                region_features=region_feats,
            )

            # Diffusion denoising loss (L_simple)
            diffusion_loss = F.mse_loss(noise_pred, noise)

            # Decode latents for perceptual / identity / geometry losses
            # (only at select timesteps to save compute)
            losses: Dict[str, torch.Tensor] = {"diffusion": diffusion_loss}

            if self.global_step % 5 == 0:
                pred_latents = self.backbone.noise_scheduler.step(
                    noise_pred, timesteps[0], noisy_latents
                ).prev_sample
                gen_images = self.backbone.decode_latents(pred_latents)

                losses["identity"] = self.identity_loss(gen_images, src_images)
                losses["perceptual"] = self.perceptual_loss(gen_images, tgt_images)
                losses["pixel"] = self.pixel_loss(gen_images, tgt_images)

                # Adversarial generator loss
                fake_logits = self.discriminator(gen_images)
                if isinstance(fake_logits, tuple):
                    fake_logits = fake_logits[0]
                losses["adv_g"] = self.adv_weight * hinge_g_loss(fake_logits)

                losses["geometry"] = self.geometry_loss(gen_images, tgt_images)

        # ── Generator backward ────────────────────────────────────────────
        g_loss = sum(losses.values())
        self.accelerator.backward(g_loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                list(self.backbone.unet.parameters()) +
                list(self.region_encoder.parameters()) +
                list(self.controlnet.parameters()),
                max_norm=1.0,
            )
        self.gen_opt.step()
        self.gen_scheduler.step()
        self.gen_opt.zero_grad()

        if self.ema_unet is not None:
            self.ema_unet.step(self.backbone.unet.parameters())

        # ── Discriminator backward ────────────────────────────────────────
        if "gen_images" in dir():  # only when decoded
            real_logits = self.discriminator(tgt_images.detach())
            fake_logits_d = self.discriminator(gen_images.detach())
            if isinstance(real_logits, tuple):
                real_logits = real_logits[0]
            if isinstance(fake_logits_d, tuple):
                fake_logits_d = fake_logits_d[0]
            d_loss = hinge_d_loss(real_logits, fake_logits_d)
            self.accelerator.backward(d_loss)
            self.disc_opt.step()
            self.disc_scheduler.step()
            self.disc_opt.zero_grad()
            losses["disc"] = d_loss

        return {k: v.detach().item() if isinstance(v, torch.Tensor) else v
                for k, v in losses.items()}

    # ── Validation loop ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation loop and compute metrics on a subset of val data."""
        self.backbone.unet.eval()
        self.region_encoder.eval()
        self.controlnet.eval()

        all_gen, all_src, all_tgt = [], [], []
        max_val_batches = 50  # cap for speed

        for i, batch in enumerate(self.val_loader):
            if i >= max_val_batches:
                break
            src = batch["source_image"].to(self.device, dtype=self.dtype)
            tgt = batch["target_image"].to(self.device, dtype=self.dtype)
            crops = {k: v.to(self.device, dtype=self.dtype)
                     for k, v in batch["source_regions"].items()}

            with self.accelerator.autocast():
                src_latents = self.backbone.encode_images(src)
                geo_out = self.geometry(F.interpolate(src, (224, 224)))
                controlnet_out = self.controlnet(
                    geo_out["depth_map"], geo_out["param_embedding"]
                )
                region_feats = self.region_encoder(crops)
                B = src.shape[0]
                prompt_embeds, pooled = self.backbone.encode_text([""] * B)
                added_cond = {"text_embeds": pooled,
                              "time_ids": torch.zeros(B, 6, device=self.device)}

                # One-step denoising at t=0 for fast eval
                noise = torch.randn_like(src_latents)
                ts = torch.zeros(B, dtype=torch.long, device=self.device)
                noisy = self.backbone.add_noise(src_latents, noise, ts)
                pred_noise = self.backbone(
                    noisy, ts, prompt_embeds, added_cond, controlnet_out, region_feats
                )
                pred_latents = src_latents - noise + pred_noise
                gen = self.backbone.decode_latents(pred_latents)

            all_gen.append(gen.cpu())
            all_src.append(src.cpu())
            all_tgt.append(tgt.cpu())

        gen_all = torch.cat(all_gen)
        src_all = torch.cat(all_src)
        tgt_all = torch.cat(all_tgt)

        # Log visual comparison
        save_path = self.output_dir / f"val_step_{self.global_step}.png"
        save_comparison_grid(src_all[:8], gen_all[:8], tgt_all[:8], save_path)

        metrics = compute_metrics(gen_all, tgt_all, src_all)

        self.backbone.unet.train()
        self.region_encoder.train()
        self.controlnet.train()
        return metrics

    # ── Main train loop ──────────────────────────────────────────────────────

    def train(self) -> None:
        """
        Main training entry point.  Runs until ``total_steps`` are completed.
        """
        tcfg = self.cfg["training"]
        total_steps: int = tcfg["total_steps"]
        save_every: int = tcfg["save_every_steps"]
        val_every: int = tcfg["validate_every_steps"]
        log_every: int = tcfg["log_every_steps"]

        self.accelerator.init_trackers(
            project_name=self.cfg["experiment"]["name"],
            config=self.cfg,
        )

        progress = tqdm(
            total=total_steps,
            initial=self.global_step,
            desc="Training",
            disable=not self.accelerator.is_local_main_process,
        )

        train_iter = iter(self.train_loader)
        self.backbone.unet.train()
        self.region_encoder.train()
        self.controlnet.train()
        self.discriminator.train()

        while self.global_step < total_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            with self.accelerator.accumulate(self.backbone.unet):
                losses = self._training_step(batch)

            if self.accelerator.sync_gradients:
                self.global_step += 1
                progress.update(1)

                if self.global_step % log_every == 0:
                    self.accelerator.log(losses, step=self.global_step)

                if self.global_step % val_every == 0:
                    metrics = self._validate()
                    self.accelerator.log(metrics, step=self.global_step)
                    self.accelerator.print(
                        f"[Step {self.global_step}] Val: {metrics}"
                    )

                if self.global_step % save_every == 0:
                    self._save_checkpoint()

        progress.close()
        self.accelerator.end_training()
        self.accelerator.print("Training complete.")
