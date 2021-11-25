import os
from glob import glob

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
train_fl = sorted(glob(os.path.join(data_dir, 'train_data', '*', '*T1acq_nu_FL.nii.gz')))
train_labels = sorted(glob(os.path.join(data_dir, 'train_data', '*', '*wmh_seg.nii.gz')))
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


def load_dataloader(root_dir):
    """
    """
    
    data_dir = os.path.join(root_dir, 'data', 'wmh_hab')
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


