from __future__ import annotations

try:
    from detectron2.config import CfgNode as CN
except Exception:  # pragma: no cover - optional dependency in anon release
    class CN(dict):
        def __getattr__(self, item):
            if item not in self:
                self[item] = CN()
            return self[item]

        def __setattr__(self, key, value):
            self[key] = value


def add_maskformer2_config(cfg):
    cfg.INPUT.DATASET_MAPPER_NAME = "irsr_sor"
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    cfg.MODEL.MASK_FORMER = CN()
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False

    cfg.MODEL.RELATION_HEAD = CN()
    cfg.MODEL.RELATION_HEAD.UNIT_NUMS = 8
    cfg.MODEL.RELATION_HEAD.LAYER_NUM = 1

    cfg.EVALUATION = CN()
    cfg.EVALUATION.DATASET = "irsr"
    cfg.EVALUATION.DATASETMODE = "test"
    cfg.EVALUATION.RESULT_THRESHOLD = 0.7
