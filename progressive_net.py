from __future__ import print_function, division
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import time
import os
import copy

import msra_cfw_faceid_loader as msra
import ferplus_reader as fplus_read
import ferplus_dataset as fplus_data
import imagenet_matconvnet_vgg_f_dag as vggf
from laplotter import LossAccPlotter


emotion_table = {'neutral'  : 0,
                 'happiness': 1,
                 'surprise' : 2,
                 'sadness'  : 3,
                 'anger'    : 4,
                 'disgust'  : 5,
                 'fear'     : 6,
                 'contempt' : 7}

# List of folders for training, validation and test.

train_folders = ['FER2013Train']
valid_folders = ['FER2013Valid']
test_folders = ['FER2013Test']


def get_args():
    parser = argparse.ArgumentParser(description='Progressive Neural Networks')
    parser.add_argument('-faces_path', default='/imatge/isagastiberri//small_msra_123/', type=str, help='path to face data')
    parser.add_argument('-fer_path', default='/imatge/isagastiberri/fer_plus/', type=str, help='path to fer')
    parser.add_argument('-weights_path', default='/imatge/isagastiberri/progressivenet/imagenet_matconvnet_vgg_f_dag.pth', type=str, help='path to fer')

    parser.add_argument('--faces_lr', dest='faces_lr', type=float, default=0.001, help='Optimizer learning rate for face network')
    parser.add_argument('--fer_lr', dest='fer_lr', type=float, default=0.0015, help='Optimizer learning rate for emotion network')

    parser.add_argument('--faces_step', dest='faces_step', type=int, default=10, help='scheduler step size for faces')
    parser.add_argument('--fer_step', dest='fer_step', type=int, default=10, help='scheduler step size for emotion')

    parser.add_argument('--faces_gamma', dest='faces_gamma', type=float, default=0.1, help='Scehduler gamma for faces')
    parser.add_argument('--fer_gamma', dest='fer_gamma', type=float, default=0.1, help='Scehduler gamma for emotion')

    parser.add_argument('--epochs', dest='epochs', type=int, default=15)

    args = parser.parse_known_args()
    return args[0]

#training
def train_model(num_tasks, models, dataloaders, dataset_sizes, criterion, optimizers, schedulers, epochs=15):
    since = time.time()
    num_tasks = num_tasks
    use_gpu = torch.cuda.is_available()
    final_outputs = [] #this is the variable to accumulate the outputs of all columns for each task
    middle_outputs = [] #this is the variable for keeping outputs of current task in each column

    #we iterate for each task
    for task_id in range(num_tasks):
        #everytime we do a new task, we empty the final outputs
        final_outputs[:] =[]

        # we now iterate for each previous column until the one of our task
        for i in range(0, task_id+1):
            #we save the weights with best results
            best_model_wts = copy.deepcopy(models[task_id][i].state_dict())
            model = models[task_id][i]
            optimizer = optimizers[i]
            scheduler = schedulers[i]
            # if it's not the column corresponding to the task, do not train
            if task_id != i:
                #this is the case for "previous" columns so we only need to pass data, not train
                num_epochs = 1
                middle_outputs[:] = []
            else:
                num_epochs = epochs
            dataloader = dataloaders[i]
            dataset_size = dataset_sizes[i]
            best_acc = 0.0
            # let's add a plotter for loss and accuracy
            plotter = LossAccPlotter()
            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        scheduler.step()
                        model.train(True)  # Set model to training mode
                    else:
                        model.train(False)  # Set model to evaluate mode

                    running_loss     = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    #tqdm shows a progression bar
                    for data in tqdm(dataloader[phase]):
                        # get the inputs
                        inputs, labels = data
                        if inputs.type() != int and labels.type() != int:
                            # wrap them in Variable
                            if use_gpu:
                                inputs = Variable(inputs.cuda())
                                labels = Variable(labels.cuda())

                            else:
                                inputs, labels = Variable(inputs), Variable(labels)

                            # zero the parameter gradients
                            optimizer.zero_grad()

                            # forward
                            outputs, middle_outputs = model(inputs, final_outputs)
                            #we save the outputs of this column in middle outputs and we have previous columns in final
                            _, preds = torch.max(outputs.data, 1)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                            # statistics
                            running_loss += loss.data[0] * inputs.size(0)
                            running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / dataset_size[phase]
                    epoch_acc = running_corrects / dataset_size[phase]

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))

                    if phase == 'train':
                        plotter.add_values(epoch,
                                       loss_train=epoch_loss, acc_train=epoch_acc, redraw = False)
                    else:
                        plotter.add_values(epoch,
                                           loss_val=epoch_loss, acc_val=epoch_acc, redraw = False)
                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())

                print()
                plotter.redraw()
            if task_id != i:
                final_outputs.append(middle_outputs)
                #we add the ouput of this column to final outputs

        plotter.save_plot('plots%d.%d.png' % (task_id, i))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        models[task_id][i].load_state_dict(best_model_wts)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60))

    return models


def main(args):

    #load the dataset
    num_tasks = 2 #how many tasks do we have?
    emotion_folder = args['fer_path']
    images_path = args['faces_path']
    num_classes_emotion = len(emotion_table)

    msra_cfw_faceid_datasets = {x: msra.MSRA_CFW_FaceIDDataset(root_dir=images_path, mode=x, validation_folds=4, test_split=0.0) for x in ['train', 'val']}
    msra_cfw_dataloaders  = {x: torch.utils.data.DataLoader(msra_cfw_faceid_datasets[x], batch_size=4, shuffle=True, num_workers=2) for x in ['train', 'val']}

    # read FER+ dataset.
    print("Loading data...")

    fer_datasets = {x: fplus_data.ferplus_Dataset(base_folder=emotion_folder, train_folders= train_folders,
                                                  valid_folders= valid_folders, mode=x, classes= emotion_table) for x in ['train', 'val']}
    fer_dataloaders  = {x: torch.utils.data.DataLoader(fer_datasets[x], batch_size=4, shuffle=True, num_workers=2)
                        for x in ['train', 'val']}

    dataset_sizes_face = {x: len(msra_cfw_faceid_datasets[x]) for x in ['train', 'val']}
    class_names_face  = msra_cfw_faceid_datasets['train'].classes
    num_classes_face = msra_cfw_faceid_datasets['train'].num_classes
    #
    print ('Num batches      : {}'.format(len(msra_cfw_dataloaders['train'])))
    print ('Dataset size     : {}'.format(dataset_sizes_face['train']))
    print ('Number of classes: {}'.format(num_classes_face))

    dataset_sizes_emotion = {x: len(fer_datasets[x]) for x in ['train', 'val']}

    print ('Num batches      : {}'.format(len(fer_dataloaders['train'])))
    print ('Dataset size     : {}'.format(dataset_sizes_emotion['train']))
    print ('Number of classes: {}'.format(num_classes_emotion))

    use_gpu = torch.cuda.is_available()

    dataloaders = [msra_cfw_dataloaders, fer_dataloaders]
    dataset_sizes = [dataset_sizes_face, dataset_sizes_emotion]
    num_classes = [num_classes_face, num_classes_emotion]

    #load the model
    weights_path = args['weights_path']
    models = []
    models_for_task = []
    for task in range(0, num_tasks):
        for col in range(0, task+1):
            model_vggf = vggf.imagenet_matconvnet_vgg_f_dag(col, weights_path)
            num_ftrs = model_vggf.fc8.in_features  # fc7 or fc8 ????
            model_vggf.fc8 = nn.Linear(num_ftrs, num_classes[task])
            models_for_task.append(model_vggf)
            print(model_vggf)
        models.append(models_for_task)

    optimizer_fer = optim.SGD(models[1][1].parameters(), lr=args['fer_lr'], momentum=0.9)
    optimizer_faces = optim.SGD(models[0][0].parameters(), lr=args['faces_lr'], momentum=0.9)
    optimizers_vggf = [optimizer_faces, optimizer_fer]

    scheduler_faces = lr_scheduler.StepLR(optimizer_faces, step_size= args['faces_step'], gamma=args['faces_gamma'])
    scheduler_fer = lr_scheduler.StepLR(optimizer_fer, step_size= args['fer_step'], gamma=args['fer_gamma'])
    schedulers_vggf = [scheduler_faces, scheduler_fer]


    if use_gpu:
        for task_id in range(0, num_tasks):
            for col in range(0, task_id+1):
                models[task_id][col].cuda()

    criterion = nn.CrossEntropyLoss()

    models = train_model(num_tasks, models, dataloaders, dataset_sizes, criterion, optimizers_vggf, schedulers_vggf, epochs=args['epochs'])

if __name__ == '__main__':
    main(vars(get_args()))
