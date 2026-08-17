from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn


class SetCriterion(nn.Module):
    """Interface-compatible loss scaffold for QAGNet/Mask2Former branch."""

    def __init__(
        self,
        cfg: Any,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: Dict[str, float],
        eos_coef: float,
        losses: List[str],
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def forward(self, outputs, targets, sal_scores, side_rank_score):
        _ = outputs, targets, sal_scores, side_rank_score
        raise RuntimeError(
            "Anonymous release placeholder: criterion and ranking-loss internals are withheld during review."
        )
