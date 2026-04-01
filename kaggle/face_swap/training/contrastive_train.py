"""
training/contrastive_train.py

Phase 1: Per-region contrastive training using NT-Xent loss.

Trains each region encoder independently on VGGFace2 triplets:
  - anchor   : Person A, pose 1
  - positive : Person A, pose 2  -> tokens should be CLOSE
  - negative : Person B          -> tokens should be FAR

Run one region at a time:
  python contrastive_train.py --region mouth --epochs 20
  python contrastive_train.py --region eyes  --epochs 20
  python contrastive_train.py --region ears  --epochs 20
  python contrastive_train.py --region nose  --epochs 20

Or run all regions sequentially:
  python contrastive_train.py --region all --epochs 20
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# ── NT-Xent Contrastive Loss ───────────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross Entropy loss (SimCLR / NT-Xent).

    For each anchor:
      - positive  = same identity, different pose
      - negatives = all other samples in batch (different identities)

    Loss pushes anchor/positive together, anchor/negative apart.

    Args:
        temperature: Scaling factor (lower = sharper, default 0.07)
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor   : (B, token_dim) — normalised
            positive : (B, token_dim) — same identity as anchor
            negative : (B, token_dim) — different identity

        Returns:
            scalar loss
        """
        # L2 normalise
        anchor   = F.normalize(anchor,   dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        B = anchor.shape[0]

        # Positive similarity: (B,)
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature

        # Negative similarity: (B, B) — each anchor vs all negatives in batch
        neg_sim = torch.matmul(anchor, negative.T) / self.temperature  # (B, B)

        # For each anchor, loss = -log( exp(pos) / (exp(pos) + sum(exp(neg))) )
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+B)
        labels = torch.zeros(B, dtype=torch.long, device=anchor.device)  # pos at index 0

        return F.cross_entropy(logits, labels)


# ── Training loop for one region ───────────────────────────────────────────────

def train_region(
    region: str,
    data_root: str,
    checkpoint_dir: str,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    max_identities: int = 1000,
    min_images: int = 10,
    num_views: int = 4,
    token_dim: int = 512,
    temperature: float = 0.07,
    num_workers: int = 2,
    resume: str = None,
    device: str = "cuda",
) -> None:

    print(f"\n{'='*60}")
    print(f"Training region encoder: {region.upper()}")
    print(f"{'='*60}")

    # ── Dataset ───────────────────────────────────────────────────────────
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from face_swap.data.vggface2_dataset import VGGFace2TripletDataset, collate_triplets
    from face_swap.models.region_encoder import RegionEncoder

    dataset = VGGFace2TripletDataset(
        root=data_root,
        target_region=region,
        max_identities=max_identities,
        min_images=min_images,
        num_views=num_views,
        precomputed_geo=False,   # set True if you have precomputed depth/normal
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_triplets,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    print(f"Dataset: {len(dataset)} identities | {len(loader)} batches/epoch")

    # ── Model ─────────────────────────────────────────────────────────────
    encoder = RegionEncoder(
        region_name=region,
        token_dim=token_dim,
        pretrained=True,
        freeze_stages=3,
        transformer_layers=2,
        transformer_heads=8,
        dropout=0.1,
    ).to(device)

    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in encoder.parameters())
    print(f"Model: {trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total params")

    # ── Loss + optimiser ──────────────────────────────────────────────────
    criterion = NTXentLoss(temperature=temperature)

    optimiser = torch.optim.AdamW(
        [p for p in encoder.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=lr * 0.1
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    start_epoch = 1
    best_loss   = float("inf")

    # ── Resume from checkpoint ────────────────────────────────────────────
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        optimiser.load_state_dict(ckpt["optimiser"])
        start_epoch = ckpt["epoch"] + 1
        best_loss   = ckpt.get("best_loss", best_loss)
        print(f"Resumed from {resume} (epoch {ckpt['epoch']})")

    # ── Training loop ─────────────────────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs + 1):
        encoder.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"[{region}] Epoch {epoch}/{epochs}")

        for batch in pbar:
            # Each crop: (B, num_views, 7, 64, 64)
            anchor_crop   = batch["anchor"][region].to(device)
            positive_crop = batch["positive"][region].to(device)
            negative_crop = batch["negative"][region].to(device)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                # Encode multi-view crops:
                # (B, num_views, 7, 64, 64) -> transformer aggregation -> (B, 1, token_dim) -> (B, token_dim)
                anchor_tok   = encoder(anchor_crop).squeeze(1)    # (B, token_dim)
                positive_tok = encoder(positive_crop).squeeze(1)  # (B, token_dim)
                negative_tok = encoder(negative_crop).squeeze(1)  # (B, token_dim)

                loss = criterion(anchor_tok, positive_tok, negative_tok)

            optimiser.zero_grad()
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(
                [p for p in encoder.parameters() if p.requires_grad],
                max_norm=1.0
            )
            scaler.step(optimiser)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(loader)
        scheduler.step()

        print(f"[{region}] Epoch {epoch}: avg_loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint every epoch
        ckpt_path = Path(checkpoint_dir) / f"{region}_epoch{epoch:02d}.pt"
        torch.save({
            "epoch":      epoch,
            "region":     region,
            "encoder":    encoder.state_dict(),
            "optimiser":  optimiser.state_dict(),
            "best_loss":  min(best_loss, avg_loss),
            "avg_loss":   avg_loss,
        }, ckpt_path)

        # Save best model separately
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = Path(checkpoint_dir) / f"{region}_best.pt"
            torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                        "best_loss": best_loss}, best_path)
            print(f"  -> Best model saved: {best_path}")

    print(f"\n[{region}] Training complete. Best loss: {best_loss:.4f}")
    return encoder


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Contrastive region encoder training")
    parser.add_argument("--region",          type=str,   default="mouth",
                        choices=["mouth", "eyes", "ears", "nose", "all"],
                        help="Region to train (or 'all' for sequential)")
    parser.add_argument("--data_root",       type=str,
                        default="/kaggle/input/vggface2/data/vggface2_train/train")
    parser.add_argument("--checkpoint_dir",  type=str,   default="/kaggle/working/checkpoints")
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--max_identities",  type=int,   default=1000)
    parser.add_argument("--min_images",      type=int,   default=10)
    parser.add_argument("--num_views",       type=int,   default=4)
    parser.add_argument("--token_dim",       type=int,   default=512)
    parser.add_argument("--temperature",     type=float, default=0.07)
    parser.add_argument("--num_workers",     type=int,   default=2)
    parser.add_argument("--resume",          type=str,   default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    regions = ["mouth", "eyes", "ears", "nose"] if args.region == "all" else [args.region]

    for region in regions:
        train_region(
            region=region,
            data_root=args.data_root,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_identities=args.max_identities,
            min_images=args.min_images,
            num_views=args.num_views,
            token_dim=args.token_dim,
            temperature=args.temperature,
            num_workers=args.num_workers,
            resume=args.resume if args.region != "all" else None,
            device=device,
        )


if __name__ == "__main__":
    main()
