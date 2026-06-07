"""
Character-level Transformer encoder — the hangman policy.

Input
-----
token_ids   : long  [B, L]   values in {PAD=0, MASK=1, a..z=2..27}
attn_mask   : long  [B, L]   1 = real token, 0 = pad
guessed_vec : float [B, 26]  1 if letter has been guessed (correct OR wrong)

Output
------
logits      : float [B, 26]  per-letter "is this letter in the word?"

Architecture
------------
The guessed-letter vector is projected to d_model and broadcast-added to
every token embedding so the encoder attends to it from all positions.
After encoding we pool with masked mean and pass through a small MLP head.
Total params with default config: ~824K — trains fast on Apple Silicon MPS.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from hangman.vocab import VOCAB_SIZE, MAX_WORD_LEN, PAD_ID


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    max_len: int = MAX_WORD_LEN
    d_model: int = 128   # ~824K params — sweet spot for MPS speed vs capacity
    n_heads: int = 4
    n_layers: int = 4
    dim_ff: int = 512
    dropout: float = 0.1


class HangmanPolicy(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg

        self.tok_embed    = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD_ID)
        self.pos_embed    = nn.Embedding(cfg.max_len, cfg.d_model)
        self.guessed_proj = nn.Linear(26, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 26),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].zero_()

    def forward(
        self,
        token_ids:   torch.Tensor,   # [B, L]
        attn_mask:   torch.Tensor,   # [B, L]  1=real, 0=pad
        guessed_vec: torch.Tensor,   # [B, 26]
    ) -> torch.Tensor:               # [B, 26]
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, L)

        x = self.tok_embed(token_ids) + self.pos_embed(pos)   # [B, L, D]
        g = self.guessed_proj(guessed_vec).unsqueeze(1)        # [B, 1, D]
        x = x + g                                              # broadcast

        # nn.Transformer convention: True = ignore this position
        key_padding_mask = attn_mask.eq(0)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, L, D]

        mask_f = attn_mask.unsqueeze(-1).float()
        denom  = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = (h * mask_f).sum(dim=1) / denom               # [B, D]

        return self.head(pooled)                                # [B, 26]

    @torch.no_grad()
    def predict_letter(
        self,
        token_ids:   torch.Tensor,
        attn_mask:   torch.Tensor,
        guessed_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Greedy letter prediction with already-guessed letters masked out."""
        logits = self.forward(token_ids, attn_mask, guessed_vec)
        logits = logits.masked_fill(guessed_vec.bool(), float("-inf"))
        return logits.argmax(dim=-1)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
