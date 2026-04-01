"""
models/cross_attention.py
IP-Adapter style cross-attention injection for facial region features.

Implements:
  - RegionCrossAttention: Custom cross-attention layer that injects
    region features (K, V) into U-Net self-attention query (Q) stream.
  - IPAdapterAttentionProcessor: Replaces default attention processor in
    diffusers with region-conditioned variant.
  - RegionAttentionInjector: Helper to attach processors to all U-Net layers.

Reference: IP-Adapter — Text Compatible Image Prompt Adapter for Text-to-Image
           Diffusion Models (Ye et al., 2023, https://arxiv.org/abs/2308.06721)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0
except ImportError:
    Attention = None  # type: ignore[assignment,misc]
    AttnProcessor2_0 = None  # type: ignore[assignment,misc]


# ─────────────────────────────────────────────────────────────────────────────
# Core cross-attention layer
# ─────────────────────────────────────────────────────────────────────────────


class RegionCrossAttention(nn.Module):
    """
    Single cross-attention layer for injecting one facial region's features.

    Given:
        Q  from the U-Net hidden state  (B, seq_q, query_dim)
        KV from the region encoder      (B, num_tokens, cross_dim)

    Outputs an attention-weighted context vector that is added (scaled) to
    the U-Net hidden state inside the attention processor.

    Args:
        query_dim (int): U-Net hidden dimension (attention Q dimension).
        cross_dim (int): Region feature dimension (projection_dim of encoder).
        heads (int): Number of attention heads.
        dim_head (int): Dimension per head.
        dropout (float): Attention dropout.
        scale (float): Output scale factor (like IP-Adapter's ``scale``).
    """

    def __init__(
        self,
        query_dim: int = 2048,
        cross_dim: int = 512,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale_factor = scale
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        region_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: U-Net query stream,   (B, seq_q, query_dim).
            region_features: Encoder features,   (B, num_tokens, cross_dim).

        Returns:
            Context vector to add to hidden_states, shape (B, seq_q, query_dim).
        """
        B, seq_q, _ = hidden_states.shape

        q = self.to_q(hidden_states)
        k = self.to_k(region_features)
        v = self.to_v(region_features)

        # Reshape to multi-head format
        q = self._reshape_heads(q)  # (B*h, seq_q, dim_head)
        k = self._reshape_heads(k)  # (B*h, num_tokens, dim_head)
        v = self._reshape_heads(v)  # (B*h, num_tokens, dim_head)

        # Scaled dot-product attention
        attn_weight = torch.bmm(q, k.transpose(-1, -2)) / (self.dim_head ** 0.5)
        attn_weight = F.softmax(attn_weight, dim=-1)
        out = torch.bmm(attn_weight, v)  # (B*h, seq_q, dim_head)

        # Merge heads
        out = out.view(B, self.heads, seq_q, self.dim_head)
        out = out.permute(0, 2, 1, 3).reshape(B, seq_q, -1)

        return self.to_out(out) * self.scale_factor

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, seq, inner_dim) → (B*heads, seq, dim_head)."""
        B, seq, _ = x.shape
        x = x.view(B, seq, self.heads, self.dim_head)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(B * self.heads, seq, self.dim_head)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Region attention (aggregates all regions)
# ─────────────────────────────────────────────────────────────────────────────


class MultiRegionCrossAttention(nn.Module):
    """
    Aggregates cross-attention from all facial regions.

    Creates one RegionCrossAttention per region and sums their output
    context vectors before adding to the U-Net hidden state.

    Args:
        query_dim (int): U-Net attention query dimension.
        cross_dim (int): Region feature projection dimension.
        regions (List[str]): Region names.
        heads (int): Attention heads.
        dim_head (int): Dimension per head.
        scale (float): Output scale for each sub-attention.
    """

    def __init__(
        self,
        query_dim: int = 2048,
        cross_dim: int = 512,
        regions: Optional[List[str]] = None,
        heads: int = 8,
        dim_head: int = 64,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        from models.region_encoder import REGIONS

        if regions is None:
            regions = REGIONS
        self.regions = regions

        self.attention_layers = nn.ModuleDict(
            {
                r: RegionCrossAttention(
                    query_dim=query_dim,
                    cross_dim=cross_dim,
                    heads=heads,
                    dim_head=dim_head,
                    scale=scale,
                )
                for r in regions
            }
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        region_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: U-Net hidden states, (B, seq_q, query_dim).
            region_features: Dict[region_name → (B, num_tokens, cross_dim)].

        Returns:
            Summed context, shape (B, seq_q, query_dim).
        """
        context = torch.zeros_like(hidden_states)
        for region, attn in self.attention_layers.items():
            if region in region_features:
                context = context + attn(hidden_states, region_features[region])
        return context


# ─────────────────────────────────────────────────────────────────────────────
# Diffusers attention processor
# ─────────────────────────────────────────────────────────────────────────────


class IPAdapterRegionAttnProcessor(nn.Module):
    """
    Custom attention processor (diffusers-compatible) that injects regional
    identity features via cross-attention alongside the standard attention.

    Replaces the default attention processor for designated U-Net layers.
    During inference, region features are optionally bypassed (scale=0).

    Args:
        query_dim (int): U-Net attention hidden dim.
        cross_dim (int): Region feature dim from encoder.
        regions (List[str]): Region names.
        heads (int): Number of heads.
        dim_head (int): Dim per head.
        scale (float): Cross-attention blend scale (1.0 = full injection).
    """

    def __init__(
        self,
        query_dim: int = 2048,
        cross_dim: int = 512,
        regions: Optional[List[str]] = None,
        heads: int = 8,
        dim_head: int = 64,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.region_attn = MultiRegionCrossAttention(
            query_dim=query_dim,
            cross_dim=cross_dim,
            regions=regions,
            heads=heads,
            dim_head=dim_head,
            scale=scale,
        )
        self._scale = scale

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value
        for attn in self.region_attn.attention_layers.values():
            attn.scale_factor = value

    def __call__(
        self,
        attn: "Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        region_features: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Diffusers-compatible ``__call__`` signature.

        Performs standard text cross-attention PLUS regional identity
        cross-attention.  The region context is added to the output.

        Args:
            attn: The Attention module (diffusers).
            hidden_states: U-Net hidden states (B, seq, dim).
            encoder_hidden_states: Text CLIP embeddings (B, 77, text_dim).
            attention_mask: Optional attention mask.
            region_features: Dict of region feature tensors.

        Returns:
            Updated hidden states (B, seq, dim).
        """
        residual = hidden_states

        # Standard text cross-attention (delegating to diffusers built-in logic)
        hidden_states = attn.group_norm(hidden_states) if hasattr(attn, "group_norm") else hidden_states
        hidden_states_std = self._standard_attn(
            attn, hidden_states, encoder_hidden_states, attention_mask
        )

        # Regional identity cross-attention
        if region_features is not None and self._scale > 0.0:
            region_ctx = self.region_attn(hidden_states, region_features)
            hidden_states_std = hidden_states_std + region_ctx

        return hidden_states_std + residual

    @staticmethod
    def _standard_attn(
        attn: "Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Minimal standard cross-/self-attention logic compatible with diffusers."""
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        q = attn.to_q(hidden_states)
        k = attn.to_k(encoder_hidden_states)
        v = attn.to_v(encoder_hidden_states)

        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)

        attn_probs = attn.get_attention_scores(q, k, attention_mask)
        out = torch.bmm(attn_probs, v)
        out = attn.batch_to_head_dim(out)
        return attn.to_out[0](out)


# ─────────────────────────────────────────────────────────────────────────────
# Injection helper
# ─────────────────────────────────────────────────────────────────────────────


class RegionAttentionInjector:
    """
    Attaches IPAdapterRegionAttnProcessor to all cross-attention layers
    in an SDXL U-Net.

    Args:
        unet (nn.Module): The SD-XL U-Net object from diffusers.
        cross_dim (int): Region feature projection dimension.
        regions (List[str]): Region names.
        scale (float): Injection scale (0 = disabled, 1 = full).
        target_layers (Optional[List[str]]): Layer name substrings to target.
                                              None = all cross-attention layers.
    """

    def __init__(
        self,
        unet: nn.Module,
        cross_dim: int = 512,
        regions: Optional[List[str]] = None,
        scale: float = 1.0,
        target_layers: Optional[List[str]] = None,
    ) -> None:
        self.unet = unet
        self.cross_dim = cross_dim
        self.regions = regions
        self.scale = scale
        self.target_layers = target_layers
        self._processors: Dict[str, IPAdapterRegionAttnProcessor] = {}

    def inject(self) -> Dict[str, IPAdapterRegionAttnProcessor]:
        """
        Walk the U-Net and replace attention processors.

        Returns:
            Dict of layer_name → installed IPAdapterRegionAttnProcessor.
        """
        for name, module in self.unet.named_modules():
            if Attention is not None and isinstance(module, Attention):
                if self.target_layers and not any(t in name for t in self.target_layers):
                    continue
                query_dim = module.to_q.in_features
                processor = IPAdapterRegionAttnProcessor(
                    query_dim=query_dim,
                    cross_dim=self.cross_dim,
                    regions=self.regions,
                    heads=module.heads,
                    dim_head=query_dim // module.heads,
                    scale=self.scale,
                )
                module.set_processor(processor)
                self._processors[name] = processor
        return self._processors

    def set_scale(self, scale: float) -> None:
        """Update injection scale for all installed processors at runtime."""
        for proc in self._processors.values():
            proc.scale = scale

    @property
    def processors(self) -> Dict[str, IPAdapterRegionAttnProcessor]:
        return self._processors
