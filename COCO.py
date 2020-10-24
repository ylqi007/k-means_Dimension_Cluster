#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Time    : 10/23/20 4:02 PM
# Author  : Shark
# Site    : 
# File    : COCO.py
# Software: PyCharm
# ===========================================================


from easydict import EasyDict as edict

__COCO = edict()
COCO = __COCO

__COCO.NAME = "COCO2017"


__COCO.Images_dir = ""
__COCO.Annos_dir = "./DATA/train2017_with_size.txt"

__COCO.LABELS = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane',
                 'bus', 'train', 'truck', 'boat', 'traffic light',
                 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                 'snowboard', 'sports ball', 'kite', 'baseball bat',
                 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable',
                 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote',
                 'keyboard', 'cell phone', 'microwave', 'oven',
                 'toaster', 'sink', 'refrigerator', 'book',
                 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

