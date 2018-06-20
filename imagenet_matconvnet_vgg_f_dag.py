import torch
import torch.nn as nn
import pickle

class Imagenet_matconvnet_vgg_f_dag(nn.Module):

    def __init__(self, task_id):
        super(Imagenet_matconvnet_vgg_f_dag, self).__init__()
        self.meta = {'mean': [122.80329895019531, 114.88525390625, 101.57212829589844],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.task_id = task_id
        self.conv1 = nn.Conv2d(3, 64, kernel_size=[11, 11], stride=(4, 4))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=[5, 5], stride=(1, 1), padding=(2, 2))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.fc6 = nn.Conv2d(256, 4096, kernel_size=[6, 6], stride=(1, 1))
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(in_features=4096, out_features=1000, bias=True)


    def forward(self, x0, previous_outputs = None):
        previous_outputs = previous_outputs
        current_outputs = []
        #if it's the first, we do not have previous outputs
        if self.task_id == 0:
            x1 = self.conv1(x0)
            current_outputs.append(x1)
            x2 = self.relu1(x1)
            x3 = self.pool1(x2)
            # make x3 be a combination of previous x3 =
            x4 = self.conv2(x3)
            current_outputs.append(x4)
            x5 = self.relu2(x4)
            x6 = self.pool2(x5)
            # make x6 be a combination of previous x6 =
            x7 = self.conv3(x6)
            current_outputs.append(x7)
            x8 = self.relu3(x7)
            # make x8 be a combination of previous x8 =
            x9 = self.conv4(x8)
            current_outputs.append(x9)
            x10 = self.relu4(x9)
            # make x10 be a combination of previous x10 =
            x11 = self.conv5(x10)
            current_outputs.append(x11)
            x12 = self.relu5(x11)
            x13 = self.pool5(x12)
            # make x13 be a combination of previous x13 =
            x14 = self.fc6(x13)
            current_outputs.append(x14)
            x15_preflatten = self.relu6(x14)
            x15 = x15_preflatten.view(x15_preflatten.size(0), -1)
            x16 = self.fc7(x15)
            current_outputs.append(x16)
            x17 = self.relu7(x16)
            x18 = self.fc8(x17)
        else:
            x1 = self.conv1(x0)
            current_outputs.append(x1)
            for i in range(0, self.task_id):
                x1 = x1 + previous_outputs[i][0]
            x2 = self.relu1(x1)
            x3 = self.pool1(x2)
            # make x3 be a combination of previous x3 =
            x4 = self.conv2(x3)
            current_outputs.append(x4)
            for i in range(0, self.task_id):
                x4 = x4 + previous_outputs[i][1]
            x5 = self.relu2(x4)
            x6 = self.pool2(x5)
            # make x6 be a combination of previous x6 =
            x7 = self.conv3(x6)
            current_outputs.append(x7)
            for i in range(0, self.task_id):
                x7 = x7 + previous_outputs[i][2]
            x8 = self.relu3(x7)
            # make x8 be a combination of previous x8 =
            x9 = self.conv4(x8)
            current_outputs.append(x9)
            for i in range(0, self.task_id):
                x9 = x9 + previous_outputs[i][3]
            x10 = self.relu4(x9)
            # make x10 be a combination of previous x10 =
            x11 = self.conv5(x10)
            current_outputs.append(x11)
            for i in range(0, self.task_id):
                x11 = x11 + previous_outputs[i][4]
            x12 = self.relu5(x11)
            x13 = self.pool5(x12)
            # make x13 be a combination of previous x13 =
            x14 = self.fc6(x13)
            current_outputs.append(x14)
            for i in range(0, self.task_id):
                x14 = x14 + previous_outputs[i][5]
            x15_preflatten = self.relu6(x14)
            x15 = x15_preflatten.view(x15_preflatten.size(0), -1)
            x16 = self.fc7(x15)
            current_outputs.append(x16)
            for i in range(0, self.task_id):
                x16 = x16 + previous_outputs[i][6]
            x17 = self.relu7(x16)
            x18 = self.fc8(x17)
        return x18, current_outputs


def imagenet_matconvnet_vgg_f_dag(task_id, weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Imagenet_matconvnet_vgg_f_dag(task_id)
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model
