import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader,Subset
import torchvision.models as models
import torch.nn.functional as F
from models import *

import os
import copy
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from util import *
import time

random_seed = 0
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)


#The argumention use for surrogate model training stage
transform_surrogate_train = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


#The argumention use for all testing set
transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def decode_train(dataset_path):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    outter_trainset = My_PoisonFolder(1, root=dataset_path + 'tiny-imagenet-200/train/', \
                                      transform=transform_surrogate_train)

    frog_trainset = My_PoisonFolder(0, root=dataset_path + 'tiny-imagenet-200/frog/', \
                                    transform=transform_surrogate_train)

    total_set = concoct_dataset(outter_trainset, frog_trainset)

    train_batch_size = 256
    outter_loader = torch.utils.data.DataLoader(total_set, batch_size=train_batch_size, shuffle=True, num_workers=16)
    other_loader = torch.utils.data.DataLoader(outter_trainset, batch_size=train_batch_size, shuffle=True,
                                               num_workers=16)

    frog_loader = torch.utils.data.DataLoader(frog_trainset, batch_size=train_batch_size, shuffle=True, num_workers=16)

    decode_model = ResNet18_201()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        decode_model = nn.DataParallel(decode_model)
    device = torch.device("cuda:0" if True else "cpu")
    decode_model = decode_model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    # outer_opt = torch.optim.RAdam(params=base_model.parameters(), lr=generating_lr_outer)
    optimizer = torch.optim.SGD(params=decode_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # time_list = []

    for epoch in range(0, 200):
        since = time.time()
        decode_model.train()
        loss_list = []
        loop = tqdm(outter_loader, total=len(outter_loader))
        for images, labels in loop:
            loop.set_description("epoch {}|{}".format(200, epoch))  # 进度条前加内容
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = decode_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            loss_list.append(float(loss.data))
            optimizer.step()


        scheduler.step()


        ave_loss = np.average(np.array(loss_list))

        time_elapsed = time.time() - since
        print('Epoch:%d, Loss: %f, time:%0.3f s' % (epoch, ave_loss, time_elapsed))

        # record time
        # time_list.append(time_elapsed)
    # Save the surrogate model
    save_path = './checkpoint_my/decode_pretrain_binary' + str(200) + '.pth'
    torch.save(decode_model.state_dict(), save_path)

    acc = d2l.evaluate_accuracy_gpu(decode_model, other_loader)

    print(f'outter_loader acc is {acc}')

if __name__ == '__main__':
    decode_train('./data/')









