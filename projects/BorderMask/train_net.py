#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader
from border_mask import add_bordermask_config
import detectron2.data.transforms as T
from border_mask import MotionBlur


def build_polyp_segm_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    if cfg.INPUT.BLUR.ENABLED:
        augs.append(MotionBlur(prob=cfg.INPUT.BLUR.Prob, kernel= cfg.INPUT.BLUR.KERNEL_SIZE))
    augs.append(T.RandomFlip())
    return augs

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def build_train_loader(cls,cfg):
        if cfg.INPUT.AUG:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=build_polyp_segm_train_aug(cfg))
        else:
            mapper = DatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_bordermask_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def dataset_register():
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("quexian_train", {}, "datasets/quexian/train.json", "datasets/quexian")
    register_coco_instances("quexian_val", {}, "datasets/quexian/val.json", "datasets/quexian")

    register_coco_instances("endocv2021_train", {}, "datasets/EndoCV2021/annotations/instances_train2017.json",
                            "datasets/EndoCV2021/train2017")
    register_coco_instances("endocv2021_val", {}, "datasets/EndoCV2021/annotations/instances_val2017.json",
                            "datasets/EndoCV2021/val2017")

    # register_coco_instances("ETIS_train", {}, "datasets/polyp/annotations/instances_train2017.json",
    #                         "datasets/polyp/train2017")
    # register_coco_instances("ETIS_val", {}, "datasets/polyp/annotations/instances_val2017.json",
    #                         "datasets/polyp/val2017")

    register_coco_instances("ETIS_train", {}, "datasets/ETIS-LaribPolypDB/annotations/instances_train2017.json",
                            "datasets/ETIS-LaribPolypDB/train2017")
    register_coco_instances("ETIS_val", {}, "datasets/ETIS-LaribPolypDB/annotations/instances_val2017.json",
                            "datasets/ETIS-LaribPolypDB/val2017")

    register_coco_instances("CVC_ClinicDB_train", {}, "datasets/CVC-ClinicDB/annotations/instances_train2017.json",
                            "datasets/CVC-ClinicDB/train2017")
    register_coco_instances("CVC_ClinicDB_val", {}, "datasets/CVC-ClinicDB/annotations/instances_val2017.json",
                            "datasets/CVC-ClinicDB/val2017")

    register_coco_instances("CVC_ColonDB_train", {}, "datasets/CVC-ColonDB/annotations/instances_train2017.json",
                            "datasets/CVC-ColonDB/train2017")
    register_coco_instances("CVC_ColonDB_val", {}, "datasets/CVC-ColonDB/annotations/instances_val2017.json",
                            "datasets/CVC-ColonDB/val2017")

    register_coco_instances("CVC_300_train", {}, "datasets/CVC-300/annotations/instances_train2017.json",
                            "datasets/CVC-300/train2017")
    register_coco_instances("CVC_300_val", {}, "datasets/CVC-300/annotations/instances_val2017.json",
                            "datasets/CVC-300/val2017")

    register_coco_instances("Kvasir_train", {}, "datasets/Kvasir/annotations/instances_train2017.json",
                            "datasets/Kvasir/train2017")
    register_coco_instances("Kvasir_val", {}, "datasets/Kvasir/annotations/instances_val2017.json",
                            "datasets/Kvasir/val2017")

def main(args):
    dataset_register()
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg) #建立模型
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
            
        ) #检测点载入，训练
        res = Trainer.test(cfg, model) #测试
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    


    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
