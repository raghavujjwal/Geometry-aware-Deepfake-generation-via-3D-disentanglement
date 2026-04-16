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

import json
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
from utils.face_crop import FaceRegionCropper, REGIONS as FACE_REGIONS
from utils.metrics import compute_metrics
from utils.visualize import save_comparison_grid
from utils.weight_loader import load_pretrained_region_encoders


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

        # EMA — U-Net is frozen so EMA is only useful for trainable components.
        # Currently disabled; the frozen U-Net preserves the SDXL generative prior
        # and the trainable ControlNet / region encoders / cross-attn converge well
        # without EMA.  If needed, EMA can be applied to self.controlnet or
        # self.region_encoder parameters instead.
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
            freeze_unet=True,   # Freeze U-Net; learning is in encoders + ControlNet + cross-attn
            use_xformers=True,
            device=str(self.device),
            dtype=self.dtype,
        )

        rec_cfg = mcfg["region_encoder"]
        self.region_encoder = FaceRegionEncoder(
            projection_dim=rec_cfg["projection_dim"],
            num_tokens=mcfg["cross_attention"]["num_tokens"],
            regions=rec_cfg["regions"],
            in_channels=rec_cfg.get("in_channels", 7),
            freeze_backbone=rec_cfg.get("freeze_backbone", True),
            num_transformer_layers=rec_cfg.get("num_transformer_layers", 2),
            num_heads=rec_cfg.get("num_heads", 8),
        )

        # Load pre-trained region encoder weights (from similarity pre-training)
        pretrained_w = rec_cfg.get("pretrained_weights", {})
        if pretrained_w:
            weight_paths = {k: v for k, v in pretrained_w.items() if v is not None}
            if weight_paths:
                load_pretrained_region_encoders(
                    self.region_encoder,
                    weight_paths=weight_paths,
                    strict=False,
                    verbose=True,
                )

        geo_cfg = mcfg["controlnet"]
        self.controlnet = GeometryControlNet(
            conditioning_channels=geo_cfg.get("conditioning_channels", 6),  # depth+normal
            internal_channels=geo_cfg.get("internal_channels", 128),
        )

        self.geometry = GeometryConditioning(
            deca_model_path=mcfg["deca_model_path"],
            deca_cfg_path=mcfg["deca_cfg_path"],
            image_size=self.cfg["data"]["image_size"],
            cache_dir=self.cfg["data"].get("geometry_cache_dir"),
            cache_key_root=self.cfg["data"].get(
                "geometry_cache_key_root",
                self.cfg["data"]["datasets"][0]["root"],
            ),
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
            ckpt = torch.load(str(Path(resume_path) / "trainable_state.pt"), map_location="cpu")
            self.accelerator.unwrap_model(self.region_encoder).load_state_dict(ckpt["region_encoder"])
            self.accelerator.unwrap_model(self.controlnet).load_state_dict(ckpt["controlnet"])
            self.accelerator.unwrap_model(self.discriminator).load_state_dict(ckpt["discriminator"])
            self.gen_opt.load_state_dict(ckpt["gen_opt"])
            self.disc_opt.load_state_dict(ckpt["disc_opt"])
            self.global_step = ckpt["global_step"]
            self.accelerator.print(f"Resumed at step {self.global_step}")

    def _save_checkpoint(self) -> None:
        ckpt_dir = self.output_dir / f"checkpoint-{self.global_step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Only save trainable components — skip frozen UNet to avoid ~16GB bloat
        unwrapped_region = self.accelerator.unwrap_model(self.region_encoder)
        unwrapped_controlnet = self.accelerator.unwrap_model(self.controlnet)
        unwrapped_disc = self.accelerator.unwrap_model(self.discriminator)

        torch.save({
            "global_step": self.global_step,
            "region_encoder": unwrapped_region.state_dict(),
            "controlnet": unwrapped_controlnet.state_dict(),
            "discriminator": unwrapped_disc.state_dict(),
            "gen_opt": self.gen_opt.state_dict(),
            "disc_opt": self.disc_opt.state_dict(),
        }, str(ckpt_dir / "trainable_state.pt"))

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
        src_paths  = batch.get("source_path", None)   # list[str] or None
        tgt_paths  = batch.get("target_path", None)
        # RGB region crops (B, 3, 64, 64) per region
        rgb_crops = {k: v.to(self.device, dtype=self.dtype)
                     for k, v in batch["source_regions"].items()}
        # Normalised bboxes (B, 4) per region  [x1/W, y1/H, x2/W, y2/H]
        region_bboxes = {k: v.to(self.device, dtype=self.dtype)
                         for k, v in batch["source_region_bboxes"].items()}

        # ── Encode to latent space ────────────────────────────────────────
        with torch.no_grad():
            src_latents = self.backbone.encode_images(src_images)
            tgt_latents = self.backbone.encode_images(tgt_images)

        # ── Geometry conditioning (source + target) ───────────────────────
        with self.accelerator.autocast():
            # Source geometry → depth + normal maps for region crops
            # Pass image_paths so GeometryConditioning can load from cache
            src_geo = self.geometry(
                F.interpolate(src_images, (224, 224)),
                return_depth=True,
                return_normal=True,
                image_paths=src_paths,
            )
            src_depth_raw = src_geo["depth_map_raw"]  # (B, 1, 512, 512)  raw scalar depth
            src_normal    = src_geo["normal_map"]      # (B, 3, 512, 512)

            # Target geometry → depth + normal maps for ControlNet
            tgt_geo = self.geometry(
                F.interpolate(tgt_images, (224, 224)),
                return_depth=True,
                return_normal=True,
                image_paths=tgt_paths,
            )
            tgt_depth  = tgt_geo["depth_map"]   # (B, 3, 512, 512)  projected for ControlNet
            tgt_normal = tgt_geo["normal_map"]  # (B, 3, 512, 512)

            # ControlNet sees target depth(3ch) ‖ target normal(3ch) = 6 channels
            # Resize to latent space (8× VAE compression) so residuals match SDXL skip sizes
            tgt_condition = torch.cat([tgt_depth, tgt_normal], dim=1)
            latent_size = tgt_images.shape[-1] // 8  # e.g. 256//8 = 32
            tgt_condition_latent = F.interpolate(
                tgt_condition.float(), (latent_size, latent_size),
                mode="bilinear", align_corners=False,
            ).to(tgt_condition.dtype)
            param_emb = tgt_geo["param_embedding"]
            controlnet_out = self.controlnet(tgt_condition_latent, param_emb)

        # ── Build 7-channel region crops [RGB(3) ‖ normal(3) ‖ depth(1)] ─
        with self.accelerator.autocast():
            # Crop source normal and depth maps at the same facial region bboxes
            src_normal_crops  = FaceRegionCropper.crop_tensor_regions(
                src_normal, region_bboxes, out_size=64
            )
            src_depth_crops = FaceRegionCropper.crop_tensor_regions(
                src_depth_raw, region_bboxes, out_size=64
            )
            # Concatenate: RGB(3) + normal(3) + depth(1) = 7 channels
            region_crops_7ch = {
                region: torch.cat(
                    [rgb_crops[region], src_normal_crops[region], src_depth_crops[region]],
                    dim=1,
                )
                for region in rgb_crops
            }

        # ── Region features ───────────────────────────────────────────────
        with self.accelerator.autocast():
            region_feats = self.region_encoder(region_crops_7ch)

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

                if self.geometry_loss.weight > 0:
                    losses["geometry"] = self.geometry_loss(gen_images, tgt_images)

        # ── Generator backward ────────────────────────────────────────────
        g_loss = sum(losses.values())
        self.accelerator.backward(g_loss)
        if self.accelerator.sync_gradients:
            # Only clip trainable parameters: cross-attn injection + region
            # encoders + ControlNet.  U-Net backbone is frozen.
            self.accelerator.clip_grad_norm_(
                self.backbone.attention_injection_parameters() +
                self.region_encoder.trainable_parameters() +
                list(self.controlnet.parameters()),
                max_norm=1.0,
            )
        self.gen_opt.step()
        self.gen_scheduler.step()
        self.gen_opt.zero_grad()

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
        max_val_batches = self.cfg["training"].get(
            "max_val_batches",
            2 if self.cfg["training"]["total_steps"] <= 10 else 50,
        )

        for i, batch in enumerate(self.val_loader):
            if i >= max_val_batches:
                break
            src = batch["source_image"].to(self.device, dtype=self.dtype)
            tgt = batch["target_image"].to(self.device, dtype=self.dtype)
            rgb_crops = {k: v.to(self.device, dtype=self.dtype)
                         for k, v in batch["source_regions"].items()}
            region_bboxes = {k: v.to(self.device, dtype=self.dtype)
                             for k, v in batch["source_region_bboxes"].items()}

            with self.accelerator.autocast():
                src_latents = self.backbone.encode_images(src)

                src_geo = self.geometry(
                    F.interpolate(src, (224, 224)),
                    return_depth=True, return_normal=True,
                )
                tgt_geo = self.geometry(
                    F.interpolate(tgt, (224, 224)),
                    return_depth=True, return_normal=True,
                )
                tgt_condition = torch.cat(
                    [tgt_geo["depth_map"], tgt_geo["normal_map"]], dim=1
                )
                # Resize conditioning to latent space (same as training step)
                latent_size = tgt.shape[-1] // 8
                tgt_condition_latent = F.interpolate(
                    tgt_condition.float(), (latent_size, latent_size),
                    mode="bilinear", align_corners=False,
                ).to(tgt_condition.dtype)
                controlnet_out = self.controlnet(
                    tgt_condition_latent, tgt_geo["param_embedding"]
                )

                src_normal_crops = FaceRegionCropper.crop_tensor_regions(
                    src_geo["normal_map"], region_bboxes, out_size=64
                )
                src_depth_crops = FaceRegionCropper.crop_tensor_regions(
                    src_geo["depth_map_raw"], region_bboxes, out_size=64
                )
                # RGB(3) + normal(3) + depth(1) = 7 channels
                region_crops_7ch = {
                    region: torch.cat(
                        [rgb_crops[region],
                         src_normal_crops[region],
                         src_depth_crops[region]],
                        dim=1,
                    )
                    for region in rgb_crops
                }
                region_feats = self.region_encoder(region_crops_7ch)
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

        self.backbone.unet.eval()   # stays in eval (frozen)
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
            config=self._tracker_config(),
        )

        progress = tqdm(
            total=total_steps,
            initial=self.global_step,
            desc="Training",
            disable=not self.accelerator.is_local_main_process,
        )

        train_iter = iter(self.train_loader)
        self.backbone.unet.eval()   # U-Net is frozen; keep in eval mode
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

                if self.global_step % save_every == 0:
                    self._save_checkpoint()

                if self.global_step % val_every == 0:
                    metrics = self._validate()
                    self.accelerator.log(metrics, step=self.global_step)
                    self.accelerator.print(
                        f"[Step {self.global_step}] Val: {metrics}"
                    )

        progress.close()
        self.accelerator.end_training()
        self.accelerator.print("Training complete.")

    def _tracker_config(self) -> Dict[str, Any]:
        """Flatten YAML config to TensorBoard hparam-compatible scalar values."""
        flat: Dict[str, Any] = {}

        def add(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    add(f"{prefix}.{key}" if prefix else str(key), item)
            elif isinstance(value, (int, float, str, bool)) or value is None:
                flat[prefix] = "" if value is None else value
            else:
                flat[prefix] = json.dumps(value, sort_keys=True)

        add("", self.cfg)
        return flat
