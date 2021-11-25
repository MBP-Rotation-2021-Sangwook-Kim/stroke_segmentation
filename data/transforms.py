import glob
import os
import pickle
import time

import matplotlib.pyplot as plt
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import compute_hausdorff_distance, compute_meandice, DiceMetric, HausdorffDistanceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
import monai.transforms as tf
from monai.utils import get_torch_version_tuple, set_determinism
import numpy as np
import pandas as pd
import torch


################################################################################
### DEFINE TRANSFORMS
################################################################################

# train transforms
train_transforms = tf.Compose([
    tf.LoadImaged(keys=['image', 'label']),
    tf.AddChanneld(keys=['image', 'label']),
    tf.Spacingd(keys=['image', 'label'], pixdim=(1., 1., 1.), diagonal=True),
    tf.ToTensord(keys=['image', 'label']),
    tf.AsDiscreted(keys=['label'], threshold_values=True),
    tf.ToNumpyd(keys=['image', 'label']),
    # tf.OneOf(transforms=oneof_transforms),
    tf.NormalizeIntensityd(keys=['image'], channel_wise=True),
    tf.RandFlipd(keys=['image', 'label'], prob=0.5),
    tf.RandCropByPosNegLabeld(
        keys=['image', 'label'],
        label_key='label',
        spatial_size=(96,)*3,
        pos=1,
        neg=1,
        num_samples=4,
        image_key='image',
        image_threshold=0
    ),
    tf.ToTensord(keys=['image', 'label']),
    tf.DeleteItemsd(keys=['image_transforms', 'label_transforms'])
])

# validation and test transforms
val_transforms = tf.Compose([
    tf.LoadImaged(keys=['image', 'label']),
    tf.AddChanneld(keys=['image', 'label']),
    tf.Spacingd(keys=['image', 'label'], pixdim=(1., 1., 1.), diagonal=True),
    tf.ToTensord(keys=['image', 'label']),
    tf.AsDiscreted(keys=['label'], threshold_values=True),
    tf.ToNumpyd(keys=['image', 'label']),
    tf.NormalizeIntensityd(keys=['image'], channel_wise=True),
    tf.ToTensord(keys=['image', 'label'])
])

