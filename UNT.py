"""
Stores the basic information about UNT dataset.
"""

from easydict import EasyDict as edict

__UNT = edict()
UNT = __UNT

__UNT.NAME = "UNT_Aerial_Dataset"
__UNT.LABELS = LABELS = ['person',
                         'boat',
                         'car',
                         'bicycle',
                         'truck',
                         'bus']
__UNT.Images_dir = "./DATA/UNT_Aerial_Dataset/train/Images/"
__UNT.Annos_dir = "./DATA/UNT_Aerial_Dataset/train/Annotations/"


# class UNT_Dataset(object):
#     def __init__(self):
#         self.NAME = "UNT_Aerial_Dataset"
#         self.LABELS = LABELS = ['person',
#                                 'boat',
#                                 'car',
#                                 'bicycle',
#                                 'truck',
#                                 'bus']
#         self.Images_dir = "./DATA/UNT_Aerial_Dataset/train/Images/"
#         self.Annos_dir = "./DATA/UNT_Aerial_Dataset/train/Annotations/"
