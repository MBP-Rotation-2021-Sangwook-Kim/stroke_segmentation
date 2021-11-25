# import argparse
import glob
import os
import pickle
import time

import matplotlib.pyplot as plt
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
# from monai.metrics import compute_meandice, compute_hausdorff_distance
from monai.metrics import compute_hausdorff_distance, compute_meandice, DiceMetric, HausdorffDistanceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNETR
import monai.transforms as tf
from monai.utils import get_torch_version_tuple, set_determinism
import numpy as np
import pandas as pd
import torch
import torchio as tio

print_config()

if get_torch_version_tuple() < (1, 6):
    raise RuntimeError('AMP feature only exists in PyTorch version greater than v1.6.')

# set AMP
amp = True

################################################################################
### ARGPARSE SETUP
################################################################################

# parser = argparse.ArgumentParser(description='Train 3D Res U-Net with contrast (compression) augmentation.')
# parser.add_argument('--prob', nargs=1, type=float,
#                     help='Global probability that the transform gets applied during augmentation.')
# args = parser.parse_args()

################################################################################
### DATA SETUP
################################################################################

# get path to $SLURM_TMPDIR
slurm_tmpdir = os.getenv('SLURM_TMPDIR')

out_dir = os.path.join(slurm_tmpdir, 'out')
print(f'Files will be saved to: {out_dir}')
# TODO create specific output dirs
# TODO create dir for images and tar?

set_determinism(seed=0)

# get data files
data_dir = os.path.join(slurm_tmpdir, 'data', 'wmh_hab')
val_list_path = os.path.join(data_dir, 'wmh_validation_subjs.txt')

print('Loading files...')

# training files (remove validation subjects after)
# loading FLAIR only. Note that FLAIR is loaded as 'image' key in dict
train_fl = sorted(glob.glob(os.path.join(data_dir, 'train_data', '*', '*T1acq_nu_FL.nii.gz')))
train_labels = sorted(glob.glob(os.path.join(data_dir, 'train_data', '*', '*wmh_seg.nii.gz')))
train_files = [
    {'image': fl_name, 'label': label_name}
    for fl_name, label_name in zip(train_fl, train_labels)
]

# validation files
val_files = []
with open(val_list_path) as f:
    val_subj_list = f.read().splitlines()
    val_files = [train_file for train_file in train_files if train_file['image'].split('/')[-2] in val_subj_list]
    train_files = [train_file for train_file in train_files if train_file['image'].split('/')[-2] not in val_subj_list]

# test files
test_fl = sorted(glob.glob(os.path.join(data_dir, 'test_data', '*', '*T1acq_nu_FL.nii.gz')))
test_labels = sorted(glob.glob(os.path.join(data_dir, 'test_data', '*', '*wmh_seg.nii.gz')))
test_files = [
    {'image': fl_name, 'label': label_name}
    for fl_name, label_name in zip(test_fl, test_labels)
]

print(f'Loaded {len(train_files)} subjects for training.')
print(f'Loaded {len(val_files)} subjects for validation.')
print(f'Loaded {len(test_files)} subjects for testing.')

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

################################################################################
### DATASET AND DATALOADERS
################################################################################

# train dataset
train_ds = CacheDataset(data=train_files, transform=train_transforms,
                        cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

# valid dataset
val_ds = CacheDataset(data=val_files, transform=val_transforms,
                      cache_rate=1.0)
val_loader = DataLoader(val_ds, batch_size=1)

# test dataset
test_ds = Dataset(data=test_files, transform=val_transforms)
test_loader = DataLoader(test_ds, batch_size=1)

################################################################################
### MODEL AND LOSS
################################################################################

device = torch.device("cuda:0")
model = UNETR(
    in_channels=1,
    out_channels=2,
    img_size=(96,)*3,
    feature_size=16,
    hidden_size=768,
    mlp_dim=768,
    num_heads=12,
    pos_embed='perceptron',
    norm_name='batch',
    conv_block=True,
    res_block=True,
    dropout_rate=0.1
).to(device)
loss_function = DiceLoss(
    include_background=True,
    to_onehot_y=True,
    softmax=True
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler() if amp else None

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal model parameters: {total_params}")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable model parameters: {total_trainable_params}\n")

################################################################################
### TRAINING LOOP
################################################################################

# general training params
epoch_num = 1000   # max epochs 500
early_stop = 125
early_stop_counter = 0
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
val_loss_values = list()
metric_values = list()
epoch_times = list()
total_start = time.time()
post_pred = tf.AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
post_label = tf.AsDiscrete(to_onehot=True, n_classes=2)
dice_metric = DiceMetric(include_background=False)
hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95)

# inference params for patch-based eval
roi_size = (96,)*3
sw_batch_size = 4

print(f'Starting training over max {epoch_num} epochs...')
for epoch in range(epoch_num):
    epoch_start = time.time()
    early_stop_counter += 1
    print("-" * 10)
    print(f"epoch {epoch + 1}/{epoch_num}")
    model.train()
    epoch_loss = 0
    step = 0
    step_start = time.time()
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        if amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}"
              f" step time: {(time.time() - step_start):.4f}")
        step_start = time.time()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)
    print(f"time consuming of epoch {epoch + 1} is: {epoch_time:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            val_loss = 0
            step = 0
            metric_sum = 0
            metric_count = 0
            for val_data in val_loader:
                step += 1
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                if amp:
                    with torch.cuda.amp.autocast():
                        val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                        # val_outputs = model(val_inputs)
                        loss = loss_function(val_outputs, val_labels)
                else:
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    # val_outputs = model(val_inputs)
                    loss = loss_function(val_outputs, val_labels)
                val_loss += loss.item()
                # val_outputs = post_pred(val_outputs)
                # val_labels = post_label(val_labels)
                dice = compute_meandice(
                    y_pred=post_pred(val_outputs[0]).unsqueeze(0),
                    y=post_label(val_labels[0]).unsqueeze(0),
                    include_background=False,
                ).item()
                # val_labels_list = decollate_batch(val_labels)
                # val_labels_convert = [
                #     post_label(val_label_tensor) for val_label_tensor in val_labels_list
                # ]
                # val_outputs_list = decollate_batch(val_outputs)
                # val_outputs_convert = [
                #     post_pred(val_output_tensor) for val_output_tensor in val_outputs_list
                # ]
                # dice_metric(y_pred=val_outputs_convert, y=val_labels_convert)
                # dice = dice_metric.aggregate().item()
                metric_count += 1
                metric_sum += dice
            val_loss /= step
            val_loss_values.append(val_loss)
            metric = metric_sum / metric_count
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(out_dir, "best_metric_model.pth"))
                print("saved new best metric model")
                early_stop_counter = 0
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
            )
            if early_stop_counter >= early_stop:
                print(f"No validation metric improvement in {early_stop} epochs. "
                      f"Early stopping triggered. Breaking training loop.")
                break
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
      f" total time: {(time.time() - total_start):.4f}")
# save loss and validation metric lists
with open(os.path.join(out_dir, "train_losses.txt"), "wb") as fp:
    pickle.dump(epoch_loss_values, fp)
with open(os.path.join(out_dir, "val_losses.txt"), "wb") as fp:
    pickle.dump(val_loss_values, fp)
with open(os.path.join(out_dir, "val_metrics.txt"), "wb") as fp:
    pickle.dump(metric_values, fp)
with open(os.path.join(out_dir, "epoch_times.txt"), "wb") as fp:
    pickle.dump(epoch_times, fp)

################################################################################
### PLOT TRAINING CURVES
################################################################################

# plot loss and validation metric
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].set_title('Epoch Average Loss')
ax[0].set_xlabel('epoch')
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
ax[0].plot(x, y, label='Training loss')
x = [val_interval * (i + 1) for i in range(len(val_loss_values))]
y = val_loss_values
ax[0].plot(x, y, label='Validation loss')
ax[0].legend()
ax[1].set_title('Val Mean Dice')
ax[1].set_xlabel('epoch')
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
ax[1].plot(x, y)
plt.savefig(os.path.join(out_dir, 'training_curves.png'), bbox_inches='tight')

################################################################################
### TEST EVAL AND PLOT PREDICTIONS
################################################################################

# evaluate on test set and plot some predictions on axial slices
model.load_state_dict(torch.load(os.path.join(out_dir, "best_metric_model.pth")))
model.eval()
df = list()   # collect data for dataframe
cols = ['subject_id', 'dice', 'hausdorff_distance']
with torch.no_grad():
    dice_sum = 0.0
    dice_count = 0
    hd_sum = 0.0
    hd_count = 0
    for i, test_data in enumerate(test_loader):
        subject_id = test_data['label_meta_dict']['filename_or_obj'][0].split('/')[-2]
        test_inputs, test_labels = (
            test_data["image"].to(device),
            test_data["label"].to(device),
        )
        if amp:
            with torch.cuda.amp.autocast():
                test_outputs = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
                # test_outputs = model(test_inputs)
        else:
            test_outputs = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
            # test_outputs = model(test_inputs)
        # plot central axial slice
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].imshow(test_data["image"][0, 0, :, :, test_data['image'].shape[-1]*5//9], cmap="gray")
        ax[0].set_title(f"image {i}")
        ax[1].imshow(test_data["label"][0, 0, :, :, test_data['label'].shape[-1]*5//9])
        ax[1].set_title(f"label {i}")
        ax[2].imshow(torch.argmax(test_outputs, dim=1).detach().cpu()[0, :, :, torch.argmax(test_outputs, dim=1).shape[-1]*5//9])
        ax[2].set_title(f"output {i}")
        plt.savefig(os.path.join(out_dir, f'test_pred_{i}.png'), bbox_inches='tight')
        plt.close()
        # calculate running metrics
        # test_outputs = post_pred(test_outputs)
        # test_labels = post_label(test_labels)
        # test_labels_list = decollate_batch(test_labels)
        # test_labels_convert = [
        #     post_label(test_label_tensor) for test_label_tensor in test_labels_list
        # ]
        # test_outputs_list = decollate_batch(test_outputs)
        # test_outputs_convert = [
        #     post_pred(test_output_tensor) for test_output_tensor in test_outputs_list
        # ]
        # dice_metric(y_pred=test_outputs_convert, y=test_labels_convert)
        # dice = dice_metric.aggregate().item()
        # hd_metric(y_pred=test_outputs_convert, y=test_labels_convert)
        # hd = hd_metric.aggregate().item()
        dice = compute_meandice(
            y_pred=post_pred(test_outputs[0]).unsqueeze(0),
            y=post_label(test_labels[0]).unsqueeze(0),
            include_background=False
        ).item()
        dice_count += 1
        dice_sum += dice
        hd = compute_hausdorff_distance(
            y_pred=post_pred(test_outputs[0]).unsqueeze(0),
            y=post_label(test_labels[0]).unsqueeze(0),
            include_background=False,
            percentile=95
        ).item()
        hd_count += 1
        hd_sum += hd
        row_entry = [subject_id, dice, hd]
        df.append(dict(zip(cols, row_entry)))
    dice = dice_sum / dice_count
    hd = hd_sum / hd_count
    print(f"Best Dice score on test set: {dice:.4f}")
    print(f"Best Hausdorff distance on test set: {hd:.4f}")
df = pd.DataFrame(df)
print('Saving test eval outputs to csv in output dir...')
df.to_csv(os.path.join(out_dir, 'test_eval.csv'))
print('Done. Finished saving files.')
