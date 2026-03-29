"""
models/backbone.py
Stable Diffusion XL U-Net backbone wrapper for face swapping.

Wraps the diffusers SDXL U-Net to:
  1. Accept optional geometry conditioning tensors (from ControlNet)
  2. Support injection of region identity features via cross-attention processors
  3. Provide convenient noise prediction and latent encoding interfaces
  4. Expose parameter groups for differential learning rates

TODO: Set SDXL_MODEL_ID to your local path or HuggingFace model ID.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from models.cross_attention import RegionAttentionInjector


# ─────────────────────────────────────────────────────────────────────────────
# Constants / defaults
# ─────────────────────────────────────────────────────────────────────────────

# TODO: Set to local path or HuggingFace model ID
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"


# ─────────────────────────────────────────────────────────────────────────────
# SDXL Backbone Wrapper
# ─────────────────────────────────────────────────────────────────────────────


class FaceSwapBackbone(nn.Module):
    """
    SDXL backbone for geometry-aware face swapping.

    Loads SDXL's U-Net + VAE + CLIP text encoders and wires in:
      - IP-Adapter style region cross-attention processors.
      - ControlNet conditioning injection support (external).

    Only the U-Net cross-attention projection layers are trained by default.
    The VAE is always kept frozen.

    Args:
        sdxl_model_id (str): HuggingFace model ID or local path for SDXL.
        vae_model_id (str): Optional better VAE (fp16-fix). If None, uses SDXL's.
        region_projection_dim (int): Dimension of region feature tokens.
        regions (List[str]): Region names to inject.
        attn_scale (float): Region cross-attention scale.
        freeze_unet (bool): If True, freeze all U-Net weights (only train attn inj.).
        use_xformers (bool): Enable xformers memory-efficient attention if available.
        device (str): Torch device.
        dtype (torch.dtype): Model dtype (torch.float16 for fp16 training).
    """

    def __init__(
        self,
        sdxl_model_id: str = SDXL_MODEL_ID,
        vae_model_id: Optional[str] = VAE_MODEL_ID,
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

        print(f"[FaceSwapBackbone] Loading SDXL from {sdxl_model_id} ...")

        # ── U-Net ────────────────────────────────────────────────────────────
        self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            sdxl_model_id,
            subfolder="unet",
            torch_dtype=dtype,
        )
        if freeze_unet:
            for p in self.unet.parameters():
                p.requires_grad_(False)

        # Enable xformers memory-efficient attention
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
            torch_dtype=dtype if vae_model_id else dtype,
        )
        for p in self.vae.parameters():
            p.requires_grad_(False)

        # ── CLIP Text Encoders (frozen) ───────────────────────────────────────
        self.tokenizer_1: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            sdxl_model_id, subfolder="tokenizer"
        )
        self.tokenizer_2: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            sdxl_model_id, subfolder="tokenizer_2"
        )
        self.text_encoder_1: CLIPTextModel = CLIPTextModel.from_pretrained(
            sdxl_model_id, subfolder="text_encoder", torch_dtype=dtype
        )
        self.text_encoder_2: CLIPTextModelWithProjection = CLIPTextModelWithProjection.from_pretrained(
            sdxl_model_id, subfolder="text_encoder_2", torch_dtype=dtype
        )
        for enc in (self.text_encoder_1, self.text_encoder_2):
            for p in enc.parameters():
                p.requires_grad_(False)

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
        """
        Encode images into VAE latent space.

        Args:
            images: (B, 3, H, W) tensors normalised to [-1, 1].

        Returns:
            Latents (B, 4, H/8, W/8) scaled by VAE scaling factor.
        """
        images = images.to(dtype=self.dtype)
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode VAE latents back to pixel space.

        Args:
            latents: (B, 4, H/8, W/8).

        Returns:
            Images (B, 3, H, W), range [-1, 1].
        """
        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents).sample
        return images

    # ── Text conditioning helpers ─────────────────────────────────────────────

    @torch.no_grad()
    def encode_text(
        self,
        prompts: List[str],
        device: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompts with dual CLIP encoders (SDXL style).

        Args:
            prompts: List of text prompt strings.
            device: Target device. Defaults to self.device_str.

        Returns:
            Tuple of:
                prompt_embeds: (B, 77, 2048) concatenated CLIP features.
                pooled_prompt_embeds: (B, 1280) from text_encoder_2.
        """
        device = device or self.device_str

        def _encode(tokenizer: CLIPTokenizer, encoder: CLIPTextModel, prompts: List[str]):
            tokens = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            return encoder(**tokens, output_hidden_states=True)

        out1 = _encode(self.tokenizer_1, self.text_encoder_1, prompts)
        out2 = _encode(self.tokenizer_2, self.text_encoder_2, prompts)

        # SDXL uses penultimate hidden state of both encoders
        h1 = out1.hidden_states[-2]   # (B, 77, 768)
        h2 = out2.hidden_states[-2]   # (B, 77, 1280)
        prompt_embeds = torch.cat([h1, h2], dim=-1)   # (B, 77, 2048)
        pooled = out2.text_embeds                     # (B, 1280)
        return prompt_embeds, pooled

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
        Predict noise residual via the SDXL U-Net.

        Args:
            noisy_latents: Noised latents (B, 4, H/8, W/8).
            timesteps: Diffusion timesteps (B,).
            encoder_hidden_states: CLIP text features (B, 77, 2048).
            added_cond_kwargs: SDXL additional conditioning
                               (e.g. {'text_embeds': ..., 'time_ids': ...}).
            controlnet_output: Optional ControlNet residuals dict with keys
                               'down_block_res_samples' and 'mid_block_res_sample'.
            region_features: Optional Dict[str → (B, T, D)] for cross-attn injection.
                             Passed through to attention processors.

        Returns:
            Predicted noise tensor (B, 4, H/8, W/8).
        """
        # Inject region features into attention processors
        if region_features is not None:
            for proc in self.injector.processors.values():
                # Store features for retrieval in __call__ of each processor
                proc._region_features = region_features

        unet_kwargs: Dict[str, Any] = dict(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
        )
        if added_cond_kwargs:
            unet_kwargs["added_cond_kwargs"] = added_cond_kwargs
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
        """
        Forward diffusion: add noise to latents according to scheduler.

        Args:
            latents: Clean latents (B, 4, H/8, W/8).
            noise: Random noise tensor, same shape as latents.
            timesteps: Timestep indices (B,).

        Returns:
            Noisy latents (B, 4, H/8, W/8).
        """
        return self.noise_scheduler.add_noise(latents, noise, timesteps)

    # ── Parameter groups for optimiser ────────────────────────────────────────

    def unet_parameters(self) -> List[nn.Parameter]:
        """Parameters of U-Net backbone (may be frozen)."""
        return list(self.unet.parameters())

    def attention_injection_parameters(self) -> List[nn.Parameter]:
        """Region cross-attention projection parameters only."""
        params = []
        for proc in self.injector.processors.values():
            params.extend(list(proc.region_attn.parameters()))
        return params

    def set_region_attn_scale(self, scale: float) -> None:
        """Adjust region injection scale at runtime."""
        self.injector.set_scale(scale)
