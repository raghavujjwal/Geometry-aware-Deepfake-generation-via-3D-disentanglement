"""
models/backbone.py
Stable Diffusion 1.5 U-Net backbone wrapper for face swapping.

Switched from SDXL to SD 1.5 for Kaggle compatibility:
  - Single CLIP text encoder (768-dim vs SDXL's dual 2048-dim)
  - UNet ~860M params vs SDXL's 2.6B → ~3x smaller
  - No added_cond_kwargs (time_ids / pooled embeds) needed
  - Fits comfortably in Kaggle T4 16GB VRAM + 30GB RAM
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from models.cross_attention import RegionAttentionInjector


# ─────────────────────────────────────────────────────────────────────────────
# Constants / defaults
# ─────────────────────────────────────────────────────────────────────────────

SD15_MODEL_ID = "runwayml/stable-diffusion-v1-5"


# ─────────────────────────────────────────────────────────────────────────────
# SD 1.5 Backbone Wrapper
# ─────────────────────────────────────────────────────────────────────────────


class FaceSwapBackbone(nn.Module):
    """
    SD 1.5 backbone for geometry-aware face swapping.

    Loads SD 1.5 U-Net + VAE + CLIP text encoder and wires in:
      - IP-Adapter style region cross-attention processors.
      - ControlNet conditioning injection support (external).

    Only the U-Net cross-attention injection layers are trained by default.
    The VAE and text encoder are always frozen.

    Args:
        sdxl_model_id (str): HuggingFace model ID for SD 1.5 (kept as param name for config compat).
        vae_model_id (str): Optional separate VAE. If None, uses model's built-in VAE.
        region_projection_dim (int): Dimension of region feature tokens.
        regions (List[str]): Region names to inject.
        attn_scale (float): Region cross-attention scale.
        freeze_unet (bool): Freeze all U-Net weights except attn injection.
        use_xformers (bool): Enable xformers memory-efficient attention if available.
        device (str): Torch device.
        dtype (torch.dtype): Model dtype.
    """

    def __init__(
        self,
        sdxl_model_id: str = SD15_MODEL_ID,
        vae_model_id: Optional[str] = None,
        region_projection_dim: int = 512,
        regions: Optional[List[str]] = None,
        attn_scale: float = 1.0,
        freeze_unet: bool = False,
        use_xformers: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.device_str = device
        self.dtype = dtype
        self._freeze_unet = freeze_unet

        print(f"[FaceSwapBackbone] Loading SD 1.5 from {sdxl_model_id} ...")

        # ── U-Net ────────────────────────────────────────────────────────────
        self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            sdxl_model_id,
            subfolder="unet",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if freeze_unet:
            for p in self.unet.parameters():
                p.requires_grad_(False)

        if use_xformers:
            try:
                self.unet.enable_xformers_memory_efficient_attention()
                print("[FaceSwapBackbone] xformers attention enabled.")
            except Exception:
                print("[FaceSwapBackbone] xformers not available, using default.")

        # ── VAE (frozen) ─────────────────────────────────────────────────────
        vae_id = vae_model_id if vae_model_id else sdxl_model_id
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            vae_id,
            subfolder="vae" if not vae_model_id else None,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        for p in self.vae.parameters():
            p.requires_grad_(False)

        # ── CLIP Text Encoder (frozen) ────────────────────────────────────────
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            sdxl_model_id, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            sdxl_model_id, subfolder="text_encoder", torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)

        # Pre-compute null embeddings and free encoder to save RAM.
        # SD 1.5 cross_attention_dim = 768.
        with torch.no_grad():
            tokens = self.tokenizer(
                [""], padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).to(device)
            out = text_encoder(**tokens, output_hidden_states=True)
            _null_embeds = out.hidden_states[-2]  # (1, 77, 768)

        self.register_buffer("_null_prompt_embeds", _null_embeds)

        del text_encoder
        torch.cuda.empty_cache()

        # ── Noise scheduler ──────────────────────────────────────────────────
        self.noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            sdxl_model_id, subfolder="scheduler"
        )

        # ── Region attention injection ────────────────────────────────────────
        self.injector = RegionAttentionInjector(
            unet=self.unet,
            cross_dim=region_projection_dim,
            regions=regions,
            scale=attn_scale,
        )
        self.injector.inject()
        print(f"[FaceSwapBackbone] Injected {len(self.injector.processors)} region attn processors.")

    # ── VAE helpers ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images (B, 3, H, W) in [-1,1] → latents (B, 4, H/8, W/8)."""
        images = images.to(dtype=self.dtype)
        posterior = self.vae.encode(images).latent_dist
        return posterior.sample() * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents (B, 4, H/8, W/8) → images (B, 3, H, W) in [-1,1]."""
        latents = latents / self.vae.config.scaling_factor
        return self.vae.decode(latents).sample

    # ── Text conditioning helpers ─────────────────────────────────────────────

    @torch.no_grad()
    def encode_text(
        self,
        prompts: List[str],
        device: Optional[str] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Return pre-computed null embeddings expanded for the batch.
        SD 1.5 does not use pooled embeds, so second return value is None.

        Returns:
            prompt_embeds: (B, 77, 768)
            pooled_embeds: None
        """
        B = len(prompts)
        return self._null_prompt_embeds.expand(B, -1, -1), None

    # ── Forward (noise prediction) ────────────────────────────────────────────

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_output: Optional[Dict[str, torch.Tensor]] = None,
        region_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Predict noise via SD 1.5 U-Net.

        Args:
            noisy_latents: (B, 4, H/8, W/8)
            timesteps: (B,)
            encoder_hidden_states: (B, 77, 768)
            added_cond_kwargs: Ignored (SDXL-only, kept for interface compat).
            controlnet_output: ControlNet residuals dict.
            region_features: Region cross-attention features.

        Returns:
            Predicted noise (B, 4, H/8, W/8).
        """
        if region_features is not None:
            for proc in self.injector.processors.values():
                proc._region_features = region_features

        unet_kwargs: Dict[str, Any] = dict(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
        )
        if controlnet_output:
            unet_kwargs["down_block_additional_residuals"] = controlnet_output.get("down_block_res_samples")
            unet_kwargs["mid_block_additional_residual"] = controlnet_output.get("mid_block_res_sample")

        return self.unet(**unet_kwargs).sample

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return self.noise_scheduler.add_noise(latents, noise, timesteps)

    # ── Parameter groups ─────────────────────────────────────────────────────

    def unet_parameters(self) -> List[nn.Parameter]:
        return list(self.unet.parameters())

    def trainable_unet_parameters(self) -> List[nn.Parameter]:
        if self._freeze_unet:
            return self.attention_injection_parameters()
        return list(self.unet.parameters())

    def attention_injection_parameters(self) -> List[nn.Parameter]:
        params = []
        for proc in self.injector.processors.values():
            params.extend(list(proc.region_attn.parameters()))
        return params

    def set_region_attn_scale(self, scale: float) -> None:
        self.injector.set_scale(scale)
