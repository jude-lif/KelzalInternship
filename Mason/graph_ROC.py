from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from scipy import special
import math
import sys
#import seaborn as sns; sns.set_theme()

#plt.ion()

INDEXTOTEST = 0

INPUT_FILE = sys.argv[1]

CONF_MATRIX_OUT = False

if len(sys.argv) >= 3:
	INDEXTOTEST = int(sys.argv[2])

if len(sys.argv) >= 4:
	if sys.argv[3] == "-M" or sys.argv[3] == "--matrix-heatmap":
		CONF_MATRIX_OUT = True

allOutputs, allLabels = None, None

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        #transforms.RandomResizedCrop(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/Shared'#'data/SW1-actors-autocropped'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

conf_matrix = None

def visualize_model(model, num_images=6):
    global allOutputs, allLabels, conf_matrix
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    conf_matrix = np.zeros((len(class_names), len(class_names)))
    conf_matrix_training = np.zeros((len(class_names), len(class_names)))
    output_matrix = None
    allLabels = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            #for label in labels:
            #    allLabels.append(label)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                #if images_so_far < num_images:
                #    images_so_far = images_so_far + 1
                #    ax = plt.subplot(num_images//2, 2, images_so_far)
                #    ax.axis('off')
                #    ax.set_title('predicted: {}, actual: {}'.format(class_names[preds[j]], class_names[labels[j]]))
                #    imshow(inputs.cpu().data[j])
                conf_matrix_training[preds[j]][labels[j]] = conf_matrix_training[preds[j]][labels[j]] + 1 
        images_so_far = 0
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            for label in labels.cpu():
                allLabels.append(label)

            outputs = model(inputs)
            if output_matrix is not None:
                output_matrix = np.concatenate((output_matrix, outputs.cpu()), axis=0)
            else:
                output_matrix = outputs.cpu()
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                if images_so_far < num_images:
                    images_so_far = images_so_far + 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}, actual: {}'.format(class_names[preds[j]], class_names[labels[j]]))
                    imshow(inputs.cpu().data[j])
                conf_matrix[preds[j]][labels[j]] = conf_matrix[preds[j]][labels[j]] + 1

                #if images_so_far == num_images:
                #    model.train(mode=was_training)
                #    return
        for index in range(len(class_names)):
            print(str(index) + ": " + class_names[index])
        columnList = []
        for column in conf_matrix_training.T:
            num_members = np.sum(column)
            columnList.append([column / num_members])
        transp = np.concatenate(columnList)
        conf_matrix_training = transp.T
        columnList = []
        for column in conf_matrix.T:
            num_members = np.sum(column)
            columnList.append([column / num_members])
        transp = np.concatenate(columnList)
        conf_matrix = transp.T
        print('################## Training Confusion Matrix ##################')
        print(conf_matrix_training)
        print('################## Testing Confusion Matrix ##################')
        print(conf_matrix)
        model.train(mode=was_training)
    #print(output_matrix)
    #print(output_matrix.shape)
    
    allOutputs = output_matrix[:,:len(class_names)]
    #print(allOutputs)
    #with open("key.txt", "w+") as outfile:
    #    for value in all_labels:
    #        outfile.write(str(value) + "  ")
    #with open("output_matrix.txt", "w+") as outfile:
    #    for row in output_matrix:
    #        for value in row:
    #            outfile.write(str(value) + "  ")
    #        outfile.write("\n")

probses = None
def ROCCurve():
  tpr = np.array([])
  fpr = np.array([])
  global allOutputs, allLabels, probses
  #m = special.softmax(axis=1)#nn.Softmax(dim=1)
  allOutputsSoft = special.softmax(allOutputs, axis=1)
  probses = []
  for row in allOutputs:
    probses.append(special.softmax(row))
  #probs = allOutputsSoft[:,8]
  probs = []
  for row in probses:
    probs.append(row[INDEXTOTEST])
  #print(allOutputsSoft)
  #print(probses)
  #print(len(probses))
  #print(probs)
  #print(allLabels)
  #print(allOutputsSoft)
  #print(allLabels[0])
  for i in range(10):
    if allLabels[0] == i:
      print("yay")
  thresholds = []
  for i in range(0, 1001):
    thresholds.append(math.pow(i / 1000, 3))
  #thresholds = [0.75]
  for threshold in thresholds:#range(0,101,1):
    #print(threshold/100)
    fpCounter = 0
    tpCounter = 0
    fnCounter = 0
    tnCounter = 0
    #maxOutput, preds = torch.max(allOutputsSoft, 1)
    #probs = allOutputsSoft[:,0]
    #print(preds.size());
    #print(probs)
    for i in range(len(probs)):
      if probs[i]>=threshold:
        #print(“greater than threshold”)
        if allLabels[i]==INDEXTOTEST:
          tpCounter += 1
          #if preds[i]==0:
          #  tpCounter+=1
          #else:
          #  fnCounter+=1
        else:
          fpCounter += 1
          #if preds[i]==0:
          #  fpCounter+=1
          #else:
          #  tnCounter+=1
      else:
        #print(“less than threshold”)
        if allLabels[i] == INDEXTOTEST:
          fnCounter+=1
        else:
          tnCounter+=1
    tpr = np.append(tpr, tpCounter/(tpCounter+fnCounter))
    fpr = np.append(fpr, fpCounter/(fpCounter+tnCounter))
    #print(fpr)
    #print(tpr)
  plt.plot(fpr,tpr)
  plt.xlim(-0.1,1.1)
  plt.ylim(-0.1,1.1)
  print(thresholds[0])
  print(thresholds[1000])


def Heatmap(conf_matrix):
	sns.heatmap(conf_matrix, annot=False)

#def eval_model(model):
#	was_training = model.training
#	model.eval()
#
#	with torch.no_grad():
#		for i, (inputs, labels) in enumerate(dataloaders['val']):
#		inputs = inputs.to(device)
#		labels = labels.to(device)
#
#		outputs = model(inputs)
#		_, preds = torch.max(outputs, 1)
		

#model_ft = models.resnet18(pretrained=True)
model_ft = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=False)

model_ft.load_state_dict(torch.load(INPUT_FILE))

#num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
#model_ft.fc = nn.Linear(num_ftrs, 4)

model_ft = model_ft.to(device)

#criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
#                       num_epochs=25)

visualize_model(model_ft)

ROCCurve()

#torch.save(model_ft.state_dict(), "model_output.pth")

#for i in range(len(allLabels)):
#	if allLabels[i] == 9:
#		print(probses[i])
#		print()

plt.show()

if CONF_MATRIX_OUT:
	import seaborn as sns; sns.set_theme(style="darkgrid")
	Heatmap(conf_matrix)
	plt.show()
