import torch.optim as optim
from tqdm import tqdm

import torchvision
from torch.utils.data import TensorDataset, Subset

from models import *

from util import *
import time
from my_model import *
import random

import numpy as np


random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

transform_normal = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
])

transform_after = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_one = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test_total = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])




def test_exp(dataset_path ='./data/', l_inf_r=8/255, poison_amount = 50):


    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)


    train_batch_size = 300

    frog_trainset = My_PoisonFolder(0, root=dataset_path + 'tiny-imagenet-200/frog/', \
                                    transform=transform_normal)

    frog_loader = torch.utils.data.DataLoader(frog_trainset, \
                                              batch_size=train_batch_size, shuffle=False, num_workers=16)


    decode = ResNet18_201()
    encode = StegaStampEncoder()

    decode = nn.DataParallel(decode)
    encode = nn.DataParallel(encode)

    decode = decode.to(device)
    encode = encode.to(device)



    save_path = './checkpoint_my/decode_pretrain_binary' + str(200) + '.pth'

    decode.load_state_dict(torch.load(save_path))


    save_path_encode = './checkpoint_my/encode_pretrain_vgg_l_inf_r_8_frogtoall_test_0-1200.pth'
    encode.load_state_dict(torch.load(save_path_encode))

    encode.eval()
    decode.eval()

    acc = evaluate_test_acc(decode, frog_loader, transform_one)
    if acc > 0.99:
        print(f'acc is {acc}, decode_model is successful load')
    else:
        print(f'acc is {acc}, fail to load decode_model')
        exit(0)

    test_acc = evaluate_encode_decode(encode, decode, frog_loader, l_inf_r, transform_after)
    if test_acc > 0.99:
        print(f'test_acc is {test_acc}, encode_model is successful load')
    else:
        print(f'test_acc is {test_acc}, fail to load encode_model')
        exit(0)

    criterion_none = nn.CrossEntropyLoss(reduction='none')

    max_index, min_index = find_max_loss_reduce(encode, decode, frog_loader, l_inf_r, criterion_none, transform_after)

    res = encode(frog_trainset[max_index][0].reshape(1, 3, 32, 32))

    ori_train = torchvision.datasets.CIFAR10(root=dataset_path, train=True, \
                                             download=True, transform=transform_normal)
    ori_test = torchvision.datasets.CIFAR10(root=dataset_path, train=False, \
                                            download=True, transform=transform_normal)

    test_batch_size = 256


    tar_label = ori_train.class_to_idx['frog']

    train_label = [get_labels(ori_train)[x] for x in range(len(get_labels(ori_train)))]
    test_label = [get_labels(ori_test)[x] for x in range(len(get_labels(ori_test)))]

    train_target_list = list(np.where(np.array(train_label) == tar_label)[0])
    test_target_list = list(np.where(np.array(test_label) == tar_label)[0])

    frog_test_set = Subset(ori_test, test_target_list)
    frog_test_loader = DataLoader(frog_test_set, batch_size=test_batch_size, shuffle=True, num_workers=16)

    other_test_list = list(np.where(np.array(test_label) != tar_label)[0])
    test_other_target = Subset(ori_test, other_test_list)


    # Poison traing
    random_poison_idx = random.sample(train_target_list, poison_amount)

    poison_train_target = poison_image(ori_train, random_poison_idx, res[0], l_inf_r, 2)

    poison_test_target = poison_image_test(test_other_target, res[0], l_inf_r, 6)

    print('Traing dataset size is:', len(poison_train_target), " Poison numbers is:", len(random_poison_idx))

    clean_train_loader = DataLoader(poison_train_target, batch_size=test_batch_size, shuffle=True, num_workers=16)

    poison_test_loader = DataLoader(poison_test_target, batch_size=test_batch_size, shuffle=True, num_workers=16)

    net = ResNet18()
    net = nn.DataParallel(net)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    time_list = []

    for epoch in range(200):
        since = time.time()
        encode.eval()
        net.train()
        loss_list = []

        loop = tqdm(clean_train_loader, total=len(clean_train_loader))
        for i, (X, y) in enumerate(loop):


            X, y = X.to(device), y.to(device)

            X = transform_after(X)

            y = y.reshape(-1)

            y_hat = net(X)

            optimizer.zero_grad()

            loss = criterion(y_hat, y)

            loss.backward()

            loss_list.append(float(loss.item()))

            optimizer.step()

        scheduler.step()
        time_elapsed = time.time() - since
        ave_loss = np.average(np.array(loss_list))
        time_list.append(time_elapsed)
        att = evaluate_test_effect(net, poison_test_loader, transform_one, tar_label)

        acc_clean = evaluate_test_acc(net, frog_test_loader, transform_one)

        print(f'epoch {epoch}, ave_loss {ave_loss}, attack_acc {att:.3f}\n'
              f'time {time_list[epoch]:.3f}s, acc_clean {acc_clean}')

if __name__ == '__main__':
    test_exp()