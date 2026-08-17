try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except Exception:
    pass

import copy
import itertools
import logging
import os
import warnings
warnings.filterwarnings("ignore")

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog, MetadataCatalog,
    build_detection_train_loader, build_detection_test_loader,
)
from detectron2.engine import (
    DefaultTrainer, default_argument_parser, default_setup, launch,
)
from detectron2.evaluation import (
    COCOEvaluator, DatasetEvaluators, verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    add_maskformer2_config,
)

from mask2former.data.datasets.register_assr import get_assr_dicts
from mask2former.data.datasets.register_irsr import get_irsr_dicts
from mask2former.data.datasets.register_sifr import get_sifr_dicts
from mask2former.data.dataset_mappers.assr_dataset_mapper import AssrDatasetMapper
from mask2former.data.dataset_mappers.irsr_dataset_mapper import IrsrDatasetMapper
from mask2former.data.dataset_mappers.sifr_dataset_mapper import SIFRdataDatasetMapper


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                f"no Evaluator for the dataset {dataset_name} with the type {evaluator_type}")
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "assr_sor":
            mapper = AssrDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "irsr_sor":
            mapper = IrsrDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "sifr_sor":
            mapper = SIFRdataDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            return build_detection_train_loader(cfg, mapper=None)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = IrsrDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {"lr": cfg.SOLVER.BASE_LR,
                    "weight_decay": cfg.SOLVER.WEIGHT_DECAY}

        norm_module_types = (
            torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm, torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm, torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()

        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if ("relative_position_bias_table" in module_param_name
                        or "absolute_pos_embed" in module_param_name):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                      and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                      and clip_norm_val > 0.0)

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)

    for d in ['train', 'test']:
        DatasetCatalog.register("assr_" + d, lambda d=d: get_assr_dicts(
            root='<PATH_TO_ASSR_DATASET>', mode=d))
        MetadataCatalog.get("assr_" + d).set(thing_classes=["salient_obj"])
        MetadataCatalog.get("assr_" + d).set(evaluator_type="coco")

        DatasetCatalog.register("irsr_" + d, lambda d=d: get_irsr_dicts(
            root='<PATH_TO_IRSR_DATASET>', mode=d))
        MetadataCatalog.get("irsr_" + d).set(thing_classes=["salient_obj"])
        MetadataCatalog.get("irsr_" + d).set(evaluator_type="coco")

        DatasetCatalog.register("sifr_" + d, lambda d=d: get_sifr_dicts(
            root='<PATH_TO_SIFR_DATASET>', mode=d))
        MetadataCatalog.get("sifr_" + d).set(thing_classes=["salient_obj"])
        MetadataCatalog.get("sifr_" + d).set(evaluator_type="coco")

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    raise RuntimeError(
        "Anonymous release placeholder: this script mirrors the public training pipeline shape, "
        "while core QAGNet execution details remain withheld during review."
    )
