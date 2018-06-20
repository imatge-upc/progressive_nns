import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
import ntpath
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image, ImageFile

from skimage.transform import resize

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
#handle error with truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)



class MSRA_CFW_FaceIDDataset(Dataset):
    """MSRA-CFW Face ID dataset."""

    def __init__(self, root_dir, mode='train', validation_folds=5, test_split = 0.1, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

        self.ima_labels = []
        self.ima_names  = []
        self.classes    = {}
        class_count     = 0

        extensions = ('.jpeg', 'jpg', '.png')

        # In MSRA-CFW images of a given person are stored in a subdir named
        # according to the person name. The subdir name is the person label 
        for dirName, subdirList, fileList in os.walk(root_dir):
            for fname in fileList:
                if fname.lower().endswith(extensions):
                    class_name = path_leaf(dirName)
                    self.ima_labels.append(class_name)
                    self.ima_names.append(dirName + '/' + fname)
                    # Update the classes dictionary so that each class has a corresponding numerical value
                    if class_name not in self.classes:
                        self.classes[class_name] = class_count
                        class_count = class_count + 1

        self.num_classes = len(self.classes)

        # Split first into train and test. This partition uses a fixed seed
        # so it does not change on successive invocations
        ima_train, ima_test, lab_train, lab_test = train_test_split(self.ima_names, self.ima_labels, shuffle=True,
                                                                    stratify=self.ima_labels)

        # Now, further split the train portion intro train and validation.
        # No fixed seed is provided, so each invocation will provide a different split useful for k-fold validation
        # Add a fixed seed for reproducibility
        tst_sz = 0.0 if validation_folds == 1 else 1.0 / float(validation_folds)
        ima_train, ima_val, lab_train, lab_val = train_test_split(ima_train, lab_train, shuffle=True,
                                                                  stratify=lab_train, test_size=tst_sz)

        if self.mode == 'train':
            self.labels = lab_train
            self.images = ima_train
        if self.mode == 'val':
            self.labels = lab_val
            self.images = ima_val
        if self.mode == 'test':
            self.labels = lab_test
            self.images = ima_test
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        try:
            x = io.imread(img_name)
            if self.transform:
                x = self.transform(x)
        #convert from array to image to be able to convert it to RGB
            x = Image.fromarray(x).convert("RGB")
            #convert image to array to be able to work with it
            x = np.asarray(x)
            x = resize(x, (224, 224))

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            if x.shape == (224, 224):
                print('converting gray to colour format')
                np.stack((x,) * 3, -1)
            x = x.transpose((2, 0, 1))  # Adjust to your input
            y = self.classes[self.labels[idx]]

            # Convert to torch FloatTensor
            xt = torch.from_numpy(x).float()  # Image
            yt = y  # Label

            return xt, yt
        except OSError:
            print(img_name)
            return None, None







