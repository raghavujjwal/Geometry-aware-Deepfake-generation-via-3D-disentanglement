"""
inference/pipeline.py
Full inference pipeline for geometry-aware face swapping.

Usage:
    pipeline = FaceSwapPipeline.from_pretrained("path/to/checkpoint")
    result = pipeline(
        source_image=src_pil,
        target_image=tgt_pil,
        num_inference_steps=50,
        guidance_scale=7.5,
    )
    result.image.save("output.png")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
import yaml
from diffusers import DDIMScheduler
from PIL import Image

from inference.postprocess import blend_face_regions, color_correction
from models.backbone import FaceSwapBackbone
from models.controlnet import GeometryControlNet
from models.geometry import GeometryConditioning
from models.region_encoder import FaceRegionEncoder
from utils.face_crop import FaceRegionCropper


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FaceSwapResult:
    """Container for a single face swap inference result."""

    image: Image.Image
    """The final swapped face PIL image."""

    raw_latents: Optional[torch.Tensor] = None
    """Raw decoded latents before blending (for debugging)."""

    depth_map: Optional[Image.Image] = None
    """DECA rendered depth map of source face."""

    region_crops: Optional[Dict[str, Image.Image]] = None
    """Extracted facial region crops."""


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class FaceSwapPipeline:
    """
    End-to-end face swapping inference pipeline.

    Runs DDIM denoising with:
      - Source identity injected via region cross-attention
      - Target pose/geometry conditioned via ControlNet depth maps
      - Optional text prompt conditioning via CLIP

    Args:
        backbone (FaceSwapBackbone): Loaded U-Net backbone.
        region_encoder (FaceRegionEncoder): Loaded region encoder.
        controlnet (GeometryControlNet): Loaded geometry ControlNet.
        geometry (GeometryConditioning): DECA geometry module.
        cropper (FaceRegionCropper): MediaPipe region detector.
        device (str): Target device.
        dtype (torch.dtype): Compute dtype.
    """

    def __init__(
        self,
        backbone: FaceSwapBackbone,
        region_encoder: FaceRegionEncoder,
        controlnet: GeometryControlNet,
        geometry: GeometryConditioning,
        cropper: FaceRegionCropper,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.backbone = backbone
        self.region_encoder = region_encoder
        self.controlnet = controlnet
        self.geometry = geometry
        self.cropper = cropper
        self.device = device
        self.dtype = dtype

        # DDIM scheduler for fast inference
        self.ddim_scheduler = DDIMScheduler.from_config(
            backbone.noise_scheduler.config
        )

        self.backbone.unet.eval()
        self.region_encoder.eval()
        self.controlnet.eval()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        config_path: Optional[str | Path] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> "FaceSwapPipeline":
        """
        Load a trained FaceSwapPipeline from a checkpoint directory.

        Args:
            checkpoint_dir: Path to saved checkpoint (from Accelerate save_state).
            config_path: Path to train_config.yaml. Defaults to
                         ``<checkpoint_dir>/../../configs/train_config.yaml``.
            device: Target device.
            dtype: Compute dtype.

        Returns:
            Loaded FaceSwapPipeline.
        """
        checkpoint_dir = Path(checkpoint_dir)
        if config_path is None:
            config_path = checkpoint_dir.parent.parent / "configs" / "train_config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        mcfg = cfg["model"]

        backbone = FaceSwapBackbone(
            sdxl_model_id=mcfg["sdxl_model_id"],
            vae_model_id=mcfg.get("vae_model_id"),
            region_projection_dim=mcfg["region_encoder"]["projection_dim"],
            regions=mcfg["region_encoder"]["regions"],
            device=device,
            dtype=dtype,
        )

        # Load fine-tuned U-Net weights
        unet_path = checkpoint_dir / "unet_ema" / "diffusion_pytorch_model.safetensors"
        if not unet_path.exists():
            unet_path = checkpoint_dir / "unet_ema" / "diffusion_pytorch_model.bin"
        if unet_path.exists():
            import safetensors.torch as st
            state = st.load_file(str(unet_path))
            backbone.unet.load_state_dict(state, strict=False)

        region_encoder = FaceRegionEncoder(
            vit_model_name=mcfg["region_encoder"]["vit_model"],
            projection_dim=mcfg["region_encoder"]["projection_dim"],
            num_tokens=mcfg["cross_attention"]["num_tokens"],
            regions=mcfg["region_encoder"]["regions"],
        )

        controlnet = GeometryControlNet(
            conditioning_channels=mcfg["controlnet"]["conditioning_channels"],
            block_out_channels=tuple(mcfg["controlnet"]["block_out_channels"]),
        )

        # Load fine-tuned weights for encoder + controlnet
        enc_ckpt = checkpoint_dir / "region_encoder.pt"
        if enc_ckpt.exists():
            region_encoder.load_state_dict(torch.load(str(enc_ckpt), map_location="cpu"))

        cn_ckpt = checkpoint_dir / "controlnet.pt"
        if cn_ckpt.exists():
            controlnet.load_state_dict(torch.load(str(cn_ckpt), map_location="cpu"))

        geometry = GeometryConditioning(
            deca_model_path=mcfg["deca_model_path"],
            deca_cfg_path=mcfg["deca_cfg_path"],
            device=device,
        )

        cropper = FaceRegionCropper()

        return cls(backbone, region_encoder, controlnet, geometry, cropper, device, dtype)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Image.Image,
        target_image: Image.Image,
        prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        controlnet_scale: float = 1.0,
        region_attn_scale: float = 1.0,
        output_size: tuple = (512, 512),
        blend_alpha: float = 0.95,
        apply_color_correction: bool = True,
        return_debug_info: bool = False,
    ) -> FaceSwapResult:
        """
        Perform face swap inference.

        Args:
            source_image: PIL source identity face.
            target_image: PIL target pose face.
            prompt: Optional text prompt for additional semantic control.
            num_inference_steps: DDIM denoising steps.
            guidance_scale: Classifier-free guidance scale.
            controlnet_scale: Geometry conditioning strength.
            region_attn_scale: Regional identity injection strength.
            output_size: Output image (W, H).
            blend_alpha: Blending alpha for face region compositing.
            apply_color_correction: Apply histogram color matching.
            return_debug_info: Include depth map and crops in result.

        Returns:
            FaceSwapResult containing the swapped image and optional debug info.
        """
        # Resize inputs to model resolution
        h, w = output_size[1], output_size[0]
        src_pil = source_image.convert("RGB").resize((w, h))
        tgt_pil = target_image.convert("RGB").resize((w, h))

        src_tensor = self._pil_to_tensor(src_pil)   # (1, 3, H, W)
        tgt_tensor = self._pil_to_tensor(tgt_pil)

        # ── Geometry conditioning from source ─────────────────────────────
        src_224 = F.interpolate(src_tensor, (224, 224))
        geo_out = self.geometry(src_224.to(self.device, self.dtype))
        depth_map = geo_out["depth_map"]
        param_emb = geo_out["param_embedding"]
        controlnet_out = self.controlnet(depth_map, param_emb, conditioning_scale=controlnet_scale)

        # ── Region features from source ───────────────────────────────────
        crops = self.cropper.crop_regions(src_pil)
        region_tensors = {}
        for k, v in crops.items():
            if v is not None:
                t = self._pil_to_tensor(v.resize((64, 64)))
                region_tensors[k] = t.to(self.device, self.dtype)
            else:
                region_tensors[k] = torch.zeros(1, 3, 64, 64, device=self.device, dtype=self.dtype)

        self.backbone.set_region_attn_scale(region_attn_scale)
        region_feats = self.region_encoder(region_tensors)

        # ── Text conditioning ─────────────────────────────────────────────
        prompts = [prompt or ""] * 1
        if guidance_scale > 1.0:
            neg_embeds, neg_pooled = self.backbone.encode_text([""])
            pos_embeds, pos_pooled = self.backbone.encode_text(prompts)
            encoder_hidden_states = torch.cat([neg_embeds, pos_embeds])
            pooled_embeds = torch.cat([neg_pooled, pos_pooled])
        else:
            encoder_hidden_states, pooled_embeds = self.backbone.encode_text(prompts)

        added_cond = {
            "text_embeds": pooled_embeds,
            "time_ids": torch.zeros(encoder_hidden_states.shape[0], 6, device=self.device),
        }

        # ── Initial noise from target latent ──────────────────────────────
        tgt_latents = self.backbone.encode_images(tgt_tensor.to(self.device, self.dtype))
        self.ddim_scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = tgt_latents

        # ── DDIM denoising loop ───────────────────────────────────────────
        for t in self.ddim_scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            ts = t.unsqueeze(0).to(self.device)

            noise_pred = self.backbone(
                noisy_latents=latent_model_input,
                timesteps=ts.expand(latent_model_input.shape[0]),
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond,
                controlnet_output=controlnet_out,
                region_features=region_feats,
            )

            if guidance_scale > 1.0:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            latents = self.ddim_scheduler.step(noise_pred, t, latents).prev_sample

        # ── Decode & postprocess ──────────────────────────────────────────
        raw_images = self.backbone.decode_latents(latents)
        raw_pil = self._tensor_to_pil(raw_images[0])

        final_image = blend_face_regions(raw_pil, tgt_pil, alpha=blend_alpha)

        if apply_color_correction:
            final_image = color_correction(final_image, tgt_pil)

        # Build result
        debug_depth = self._tensor_to_pil(depth_map[0]) if return_debug_info else None
        debug_crops = {k: v for k, v in crops.items() if return_debug_info and v is not None}

        return FaceSwapResult(
            image=final_image,
            raw_latents=latents if return_debug_info else None,
            depth_map=debug_depth,
            region_crops=debug_crops if debug_crops else None,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL → (1, 3, H, W) float tensor in [-1, 1]."""
        import torchvision.transforms.functional as TF
        t = TF.to_tensor(img)  # [0, 1]
        t = t * 2.0 - 1.0       # [-1, 1]
        return t.unsqueeze(0)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert (3, H, W) tensor in [-1, 1] → PIL."""
        import torchvision.transforms.functional as TF
        tensor = (tensor.clamp(-1.0, 1.0) + 1.0) / 2.0
        return TF.to_pil_image(tensor.cpu().float())
