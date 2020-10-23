"""
Stores the basic information about VOC.
"""

from easydict import EasyDict as edict

__VOC = edict()
VOC = __VOC

__VOC.NAME = "VOC2007"
__VOC.LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
__VOC.Images_dir = "./DATA/VOC2007/train/JPEGImages/"  # Since this is a dir, ended with `/`
__VOC.Annos_dir = "./DATA/VOC2007/train/Annotations/"

