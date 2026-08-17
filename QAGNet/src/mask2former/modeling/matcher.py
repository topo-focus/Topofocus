from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn


class HungarianMatcher(nn.Module):
    """Interface-compatible placeholder for assignment matching."""

    def __init__(self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0, num_points: int = 0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        _ = outputs, targets
        raise RuntimeError(
            "Anonymous release placeholder: Hungarian matching implementation is withheld during review."
        )

    def __repr__(self) -> str:
        return (
            f"HungarianMatcher(cost_class={self.cost_class}, "
            f"cost_mask={self.cost_mask}, cost_dice={self.cost_dice}, num_points={self.num_points})"
        )
