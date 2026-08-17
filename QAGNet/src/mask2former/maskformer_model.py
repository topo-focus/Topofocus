from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher


class MaskFormer(nn.Module):
    """Anonymous-release skeleton of QAGNet/Mask2Former meta-architecture."""

    def __init__(self, backbone: nn.Module, sem_seg_head: nn.Module, criterion: nn.Module, num_queries: int = 100):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries

    @classmethod
    def from_config(cls, cfg: Any) -> Dict[str, Any]:
        matcher = HungarianMatcher(
            cost_class=float(getattr(cfg.MODEL.MASK_FORMER, "CLASS_WEIGHT", 1.0)),
            cost_mask=float(getattr(cfg.MODEL.MASK_FORMER, "MASK_WEIGHT", 20.0)),
            cost_dice=float(getattr(cfg.MODEL.MASK_FORMER, "DICE_WEIGHT", 1.0)),
            num_points=112 * 112,
        )
        criterion = SetCriterion(
            cfg=cfg,
            num_classes=1,
            matcher=matcher,
            weight_dict={"loss_ce": 1.0, "loss_mask": 20.0, "loss_dice": 1.0},
            eos_coef=0.1,
            losses=["labels", "masks", "ranks"],
            num_points=112 * 112,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
        )
        return {
            "backbone": nn.Identity(),
            "sem_seg_head": nn.Identity(),
            "criterion": criterion,
            "num_queries": int(getattr(cfg.MODEL.MASK_FORMER, "NUM_OBJECT_QUERIES", 100)),
        }

    def forward(self, batched_inputs: List[Dict[str, Any]]):
        _ = batched_inputs
        raise RuntimeError(
            "Anonymous release placeholder: MaskFormer forward pass and post-processing internals are withheld."
        )
