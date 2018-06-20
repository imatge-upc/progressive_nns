# from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
import ntpath
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image, ImageFile
import ferplus_reader as fplus_read
from skimage.transform import resize
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# handle error with truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


class ferplus_Dataset(Dataset):
    """MSRA-CFW Face ID dataset."""

    def __init__(self, base_folder, train_folders, valid_folders, mode, classes, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.base_folder = base_folder
        self.train_folders = train_folders
        self.mode = mode
        self.transform = transform
        self.classes= classes
        self.num_classes = len(classes)
        extensions = ('.jpeg', 'jpg', '.png')

        params = fplus_read.FERPlusParameters(self.num_classes, 240, 240, training_mode="majority", shuffle = True)

        train_data_reader = fplus_read.FERPlusReader.create(base_folder, train_folders, "label.csv", params)
        val_data_reader = fplus_read.FERPlusReader.create(base_folder, valid_folders, "label.csv", params)

        img_train, label_train = train_data_reader.get_image_label()
        img_val, label_val = val_data_reader.get_image_label()
        # Now, further split the train portion intro train and validation.
        # No fixed seed is provided, so each invocation will provide a different split useful for k-fold validation
        # Add a fixed seed for reproducibility

        if self.mode == 'train':
            self.labels = label_train
            self.images = img_train
        if self.mode == 'val':
            self.labels = label_val
            self.images = img_val
        if self.mode == 'test':
            self.labels = lab_test
            self.images = ima_test

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # extensions = ('.jpeg', 'jpg', '.png', '.gif')
        img_name = self.images[idx]
        try:
            x = io.imread(img_name)
            if self.transform:
                x = self.transform(x)

            # print(x.shape)
            # CHANGE THIS!!!! Normalize to the network input dimension!!
            x = Image.fromarray(x).convert("RGB")
            x = np.asarray(x)
            x = resize(x, (224, 224))

            # Probably, you should substract dataset mean and variace before further processing!!

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            if x.shape == (224, 224):
                print('converting gray to colour format')
                np.stack((x,) * 3, -1)

            x = x.transpose((2, 0, 1))  # Adjust to your input
            y = self.labels[idx]

            # Convert to torch FloatTensor
            xt = torch.from_numpy(x).float()  # Image
            yt = y  # Label

            return xt, yt
        except OSError:
            print(img_name)
            return None, None







