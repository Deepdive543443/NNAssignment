import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from tqdm import tqdm
import torchvision.transforms as tf # Not the TensorFlow
import config
from matplotlib import pyplot as plt

#Normalization and augmentation from torchvision
tranform = tf.Compose([
        tf.ToTensor(),
        tf.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

tranform_aug = tf.Compose([
        tf.ToTensor(),
        tf.RandomHorizontalFlip(p=0.5),
        tf.RandomRotation(degrees=(0, 30)),
        tf.RandomResizedCrop(size=(32, 32)),
        tf.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

tranform_aug_strong = tf.Compose([
        tf.ToTensor(),
        tf.RandomHorizontalFlip(p=0.5),
        tf.RandomRotation(degrees=(0, 30)),
        tf.RandomResizedCrop(size=(32, 32)),
        tf.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

def mix_up(inputs, labels):
    '''
    Perform a mix up between each image in a mini-batch

    :param inputs: Images inputs in torch tensor
    :param labels: Images label in integer
    :return:
    '''
    #Generate the random mix up factors in half of the shape of input
    b, c, h, w = inputs.shape
    epsilon = torch.Tensor(np.random.beta(0.2,0.2,(int(b/2),1)))
    # epsilon = torch.rand(int(b/2),1)
    epsilon_map = epsilon.repeat(1,int(3*32*32)).reshape(-1,c,h,w)

    labels_onehot = torch.zeros(b, 10)
    labels_onehot[torch.arange(labels.shape[0]).long(), labels] = 1

    inputs = inputs[:int(b/2):] * epsilon_map + inputs[int(b/2)::] * (1 - epsilon_map)
    labels_onehot = labels_onehot[:int(b / 2):] * epsilon + labels_onehot[int(b / 2)::] * (1 - epsilon)
    return inputs, labels_onehot

def mix_up_training(model, optim, loader, epoch, non_linearity = "sigmoid"):
    '''
    Perform a training with mix up on an epoch

    :param model: model to train
    :param optim: optimizer of model
    :param loader: Dataloder of dataset
    :param epoch: Current epoch
    :return: cost:
    '''
    loop = tqdm(loader, leave=False)
    loss_fn = nn.BCELoss()

    cost = 0
    items = 0
    for inputs, labels in loop:
        inputs, labels = mix_up(inputs, labels)

        inputs = inputs.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        outputs = torch.sigmoid(model(inputs)) if non_linearity != "softmax" else torch.softmax(model(inputs),dim=1)

        loss = loss_fn(outputs, labels)

        optim.zero_grad()
        loss.backward()
        optim.step()

        # true, length = accuracy(outputs, labels)
        # trues += true
        # lengths += length
        loop.set_description(f"Epoch:[{epoch + 1}]")
        loop.set_postfix(loss=loss.item())
        cost += loss.item()
        items += inputs.shape[0]

    return  cost / items


def train(model, optim, loader, epoch, scheduler, lr_decay=False):
    '''
    Perform a training on an epoch

    :param model: model to train
    :param optim: optimizer of model
    :param loader: Dataloder of dataset
    :param epoch: Current epoch
    :return: cost:
    '''
    loop = tqdm(loader,leave=False)
    loss_fn = nn.CrossEntropyLoss()
    for param_group in optim.param_groups:
        lr = param_group['lr']

    # Accuracy tracking
    trues = 0
    lengths = 0

    #Loss tracking
    cost = 0
    items = 0
    for inputs, labels in loop:
        inputs = inputs.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        optim.zero_grad()
        loss.backward()
        optim.step()

        true, length = accuracy(outputs, labels)
        trues += true
        lengths += length
        loop.set_description(f"Epoch:[{epoch + 1}]")
        loop.set_postfix(loss=loss.item(), accuracy=(trues / lengths).item(), lr=lr)
        cost += loss.item()
        items += inputs.shape[0]

    if lr_decay:
        scheduler.step()

    return (trues/lengths).item(), cost / items

def test(model, test_loader, epoch, mix_up = False):
    '''
    Calculate the loss and accuracy on test set

    :param model: model to test
    :param test_loader: Dataloader of test set
    :param epoch: current epoch
    :return: Accuracy of testing
    '''
    loop = tqdm(test_loader,leave=True) # Progress bar
    loss_fn = nn.CrossEntropyLoss() if not mix_up else nn.BCEWithLogitsLoss() #Loss function
    # Accuracy tracking
    trues=0
    lengths=0
    #Loss tracking
    cost=0
    items=0


    with torch.no_grad(): # No grad because we don't need to perform backpropagation
        for inputs, labels in loop:
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)

            true, length = accuracy(outputs, labels)
            trues += true
            lengths += length

            loop.set_description(f"Testing:[{epoch + 1}]")
            loop.set_postfix(loss=loss.item(), accuracy=(trues/lengths).item())
            cost += loss.item()
            items += inputs.shape[0]
    return (trues/lengths).item(), cost / items

def KFold_validation(model, dataset, val_epochs=10, shuffle=True, split=5, lr=config.LEARNING_RATE, batch_size = config.BATCH_SIZE, mix_up_activation=False, weight_decay=0):
    #Train the model on training set, then calculate the accuracy and cost on test set.
    # Ref: https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md

    # weight = model.state_dict()
    kfold = KFold(n_splits=split, shuffle=shuffle)
    acc = []
    start_time = time.time()
    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        model_copy = copy.deepcopy(model)
        optimizer = optim.Adam(model_copy.parameters(), lr=lr, weight_decay=weight_decay)
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

        for epoch in range(val_epochs):
            train(model_copy, optimizer, train_loader, epoch, scheduler=None) if not mix_up_activation else mix_up_training(model_copy, optimizer, train_loader, epoch, softmax = mix_up_activation)
        acc_batch, _ = test(model_copy, test_loader, fold)
        acc.append(acc_batch)
    return acc, time.time() - start_time

def accuracy(outputs, labels):
    #Use argmax and mask to predict the accuracy of
    T, L = 0, 0
    with torch.no_grad():
        mask = torch.zeros_like(labels).to(config.DEVICE)
        outputs = torch.argmax(outputs, dim=1)
        mask[labels == outputs] = 1
        T += torch.sum(mask)
        L += labels.shape[0]
    return T, L

def draw_loss_acc_curve(model, loader, test_loader, title, epochs=100, lr=config.LEARNING_RATE, lr_decay=False, milestone=[20,80], gamma=0.1, **kwargs):
    '''
    Train the model, plot and save the curve of loss and Acc

    :param model: PyTorch model
    :param loader: training data loader
    :param test_loader: test data loader
    :param title: title of graph
    :param epochs: show the plot in range of epoch
    :param lr: learning rate
    :param lr_decay: use learning rate decay or not
    :param milestone: Milestone of learning rate scheduler
    :param gamma: gamma of learning rate scheduler
    :param kwargs: keyword argument of weight decay
    :return:
    '''
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    optimizer = optim.Adam(model.parameters(), lr=lr ,**kwargs) if not lr_decay else optim.SGD(model.parameters(), lr=lr ,**kwargs)
    scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=gamma)
    for epoch in range(epochs):
        acc, loss = train(model, optimizer, loader, epoch, scheduler, lr_decay)
        train_loss.append(loss)
        train_acc.append(acc)
        acc, loss = test(model, test_loader, epoch)
        test_loss.append(loss)
        test_acc.append(acc)

    # Loss curve plotting
    plt.plot(train_loss, label="train_loss")
    plt.plot(test_loss, label="test_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss per item")
    plt.title(title+" loss curve")
    plt.legend()
    plt.show()
    # Accuracy curve plotting
    plt.plot(train_acc, label="train_acc")
    plt.plot(test_acc, label="test_acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(title+" accuracy curve")
    plt.legend()
    plt.show()
    torch.save((train_loss, test_loss, train_acc, test_acc), "result/"+title+".tar")

def draw_max_acc(list_acc, tag,coor_factor=1):
    '''
    draw a pointer of the location of maximum accuracy
    '''

    x = np.argmax(list_acc)
    y = max(list_acc)
    text = f"Epoch:{x} Acc:{100 * y:.2f}% {tag}"
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords='data',
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    print(x ,y)
    plt.annotate(text, xy=(x, y), xytext=(x+(60 * coor_factor), y+(0.1 * coor_factor)), **kw)

def draw_early_stop(list_acc, tag,coor_factor=1):
    '''
    draw a pointer of location of minimum testing loss, as known as early stop
    '''
    x = np.argmin(list_acc)
    y = min(list_acc)
    text = f"Epoch:{x} Loss:{y:.4f} {tag}"
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=120")
    kw = dict(xycoords='data', textcoords='data',
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    plt.annotate(text, xy=(x, y), xytext=(x + (60 * coor_factor), y - (0.001 * coor_factor)), **kw)
