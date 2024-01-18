import logging
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2 as cv
import torch.nn as nn
from collections import OrderedDict
import torch.utils.data as data
from PIL import Image
import os
from d2l import torch as d2l
import random



random_seed = 0
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


device = torch.device("cuda:0" if True else "cpu")

np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

def attack_img_cuda(img, encode):
    img = img.to(device)
    encode = encode.to(device)
    res = encode(img)
    att = res + img
    return att, res


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)

    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1/batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)





def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 and of radius xi
    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/torch.linalg.norm(v.flatten(1)))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == 3:
        v = torch.sign(v) * torch.minimum(abs(v), torch.tensor(xi))
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')
    return v

def get_dataset_index(target_path,target_label):

    all_content=os.listdir(target_path)

    lab_count = 0
    pass_file = 0
    target_len = 0
    for content in all_content:
        files_name = os.listdir(target_path+content)
        if lab_count == target_label:
            target_len += len(files_name)
            target_list = list(range(pass_file,pass_file+target_len))
        pass_file += len(files_name)
        lab_count += 1
    non_target_list = list(set(list(range(0,pass_file))) - set(target_list))
    return target_list,non_target_list

class my_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices,labels):
        self.dataset = dataset
        self.indices = indices
        labels_hold = torch.ones(len(dataset)).type(torch.long) *300 #( some number not present in the #labels just to make sure
        labels_hold[self.indices] = labels 
        self.labels = labels_hold
    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        label = self.labels[self.indices[idx]]
        return (image, label)

    def __len__(self):
        return len(self.indices)

class my_tar_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices,labels):
        self.dataset = dataset
        self.indices = indices
        self.labels = labels

    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]

        return (image, self.labels)

    def __len__(self):
        return len(self.indices)
        


class poison_label(Dataset):
    def __init__(self, dataset,indices,target):
        self.dataset = dataset
        self.indices = indices
        self.target = target

    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        return (image, self.target)

    def __len__(self):
        return len(self.dataset)

    

class get_labels(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx][1]

    def __len__(self):
        return len(self.dataset)
    
def load_pth(input_model,load_file_path):
    loaded_dict = torch.load(load_file_path)
    new_state_dict = OrderedDict()
    for k, v in loaded_dict.items():
        name = k[7:]
        new_state_dict[name] = v 

    input_model.load_state_dict(new_state_dict)
    input_model = input_model.cuda()
    return input_model

class concoct_dataset(torch.utils.data.Dataset):
    def __init__(self, target_dataset,outter_dataset):
        self.idataset = target_dataset
        self.odataset = outter_dataset

    def __getitem__(self, idx):
        if idx < len(self.odataset):
            img = self.odataset[idx][0]
            labels = self.odataset[idx][1]
        else:
            img = self.idataset[idx-len(self.odataset)][0]
            #labels = torch.tensor(len(self.odataset.classes),dtype=torch.long)
            labels = len(self.odataset.classes)
        #label = self.dataset[idx][1]
        return (img,labels)

    def __len__(self):
        return len(self.idataset)+len(self.odataset)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]

    classes.sort()

    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:

            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class My_ImageFolder(data.Dataset):

    def __init__(self, ratio, root, transform=None, target_transform=None,
                 loader=default_loader):
        # TODO
        # 1. Initialize file path or list of file names.

        classes, class_to_idx = find_classes(root)

        imgs = make_dataset(root, class_to_idx, ratio)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).


        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        # return the total size of your dataset.
        return len(self.imgs)


def find_classes_p(dir, num):



    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]

    classes.sort()

    class_to_idx = {classes[i]: (i + 1) % num for i in range(len(classes))}
    return classes, class_to_idx


class My_PoisonFolder(data.Dataset):

    def __init__(self, my_target, root, transform=None,
                 loader=default_loader):
        # TODO
        # 1. Initialize file path or list of file names.


        classes, class_to_idx = find_classes_p(root, 2)

        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform

        self.my_target = my_target
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).


        path, target = self.imgs[index]
        target = self.my_target

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        # return the total size of your dataset.
        return len(self.imgs)


def evaluate_data(net, data_loader):
    metric = d2l.Accumulator(2)
    for images, labels in data_loader:
        images, labels = images.cuda(), labels.cuda()

        outputs = net(images)

        metric.add(d2l.accuracy(outputs, labels), images.shape[0])

    test_acc = metric[0] / metric[1]

    return test_acc


def evaluate_test_acc(net, loader, transform_after):

    net.eval()
    net = net.to(device)
    metric = d2l.Accumulator(2)
    for img, label in loader:
        img = transform_after(img)
        label = label.to(device)
        y_hat = net(img)

        with torch.no_grad():
            metric.add(d2l.accuracy(y_hat, label), label.numel())
    return metric[0] / metric[1]


def evaluate_encode_decode(encode, decode, data_loader, l_inf_r, transform_after):
    encode = encode.to(device)
    metric = d2l.Accumulator(2)
    for img_true, label in data_loader:
        img_true, label = img_true.to(device), label.to(device)
        res = encode(img_true)

        res = torch.clamp(res, -l_inf_r, l_inf_r)

        finish_img = res + img_true

        calmp_finish = torch.clamp(finish_img, 0, 1)

        trans_finish = transform_after(calmp_finish)
        y_hat = decode(trans_finish)
        y_att = torch.zeros_like(label) + 1
        with torch.no_grad():
            metric.add(d2l.accuracy(y_hat, y_att), y_att.numel())
    return metric[0] / metric[1]


def find_max_loss_reduce(encode, decode, data_loader, l_inf_r, criterion_none, transform_after):
    encode.eval()
    decode.eval()
    encode = encode.to(device)
    decode = decode.to(device)
    loss_img = []
    loss_encode = []

    for X, y in data_loader:
        y = y.to(device)
        X = X.to(device)

        label = torch.ones_like(y).cuda()

        ori_hat = decode(transform_after(X))
        loss_ori = criterion_none(ori_hat, label.reshape(-1))
        loss_img.extend(list(loss_ori.reshape(-1).tolist()))

        res = encode(X)
        res = torch.clamp(res, -l_inf_r, l_inf_r)

        finish_img = res + X

        calmp_finish = torch.clamp(finish_img, 0, 1)
        calmp_finish = transform_after(calmp_finish)

        y_hat = decode(calmp_finish)
        loss_ca = criterion_none(y_hat, label)
        loss_encode.extend(loss_ca.reshape(-1).tolist())

    c = [loss_img[i] - loss_encode[i] for i in range(len(loss_img))]
    max_index = c.index(max(c))
    min_index = c.index(min(c))
    return max_index, min_index


class poison_image(Dataset):

    def __init__(self, dataset, indices, res, l_inf_r, times):
        self.dataset = dataset
        self.indices = indices
        self.res = res.cpu()
        self.times = int(times)
        self.l_inf_r = l_inf_r

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        image_2 = image.reshape(-1, 3, 32, 32)
        image_2 = image_2.cpu()
        if idx in self.indices:
            res = torch.clamp(self.res, -self.times * self.l_inf_r, self.times * self.l_inf_r)
            finish_image = res + image_2
            image = torch.clamp(finish_image, 0, 1)
            image = image.reshape(3, 32, 32)
        label = self.dataset[idx][1]

        return (image.detach(), label)

    def __len__(self):
        return len(self.dataset)


class poison_image_test(Dataset):

    def __init__(self, dataset, res, l_inf_r, times):
        self.dataset = dataset

        self.res = res.cpu()
        self.times = times
        self.l_inf_r = l_inf_r

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        image_2 = image.reshape(-1, 3, 32, 32)
        image_2 = image_2.cpu()
        res = torch.clamp(self.res, -self.times * self.l_inf_r, self.times * self.l_inf_r)
        finish_image = res + image_2
        image = torch.clamp(finish_image, 0, 1)
        image = image.reshape(3, 32, 32)
        label = self.dataset[idx][1]

        return (image.detach(), label)

    def __len__(self):
        return len(self.dataset)


class poison_image_encode(Dataset):

    def __init__(self, dataset, indices, encode, times, l_inf_r):
        self.dataset = dataset
        self.indices = indices
        self.encode = encode.cpu()
        self.times = times
        self.l_inf_r = l_inf_r

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        image_2 = image.reshape(-1, 3, 32, 32)
        image_2 = image_2.cpu()
        if idx in self.indices:
            res = self.encode(image_2)
            res = torch.clamp(res, -self.times * self.l_inf_r, self.times * self.l_inf_r)
            finish_image = res + image_2
            image = torch.clamp(finish_image, 0, 1)
            image = image.reshape(3, 32, 32)
        label = self.dataset[idx][1]

        return (image.detach(), label)

    def __len__(self):
        return len(self.dataset)


class poison_image_encode_test(Dataset):

    def __init__(self, dataset, encode, times, l_inf_r):
        self.dataset = dataset

        self.encode = encode.cpu()

        self.times = times

        self.l_inf_r = l_inf_r

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        image_2 = image.reshape(-1, 3, 32, 32)
        image_2 = image_2.cpu()
        res = self.encode(image_2)
        res = torch.clamp(res, -self.times * self.l_inf_r, self.times * self.l_inf_r)
        finish_image = res + image_2
        image = torch.clamp(finish_image, 0, 1)
        image = image.reshape(3, 32, 32)
        label = self.dataset[idx][1]

        return (image.detach(), label)

    def __len__(self):
        return len(self.dataset)

def evaluate_test_effect(net, loader, transform_Normalize, tar):
    net.eval()
    net = net.to(device)
    metric = d2l.Accumulator(2)
    for img, label in loader:
        img = transform_Normalize(img)
        label = label.to(device)
        y_hat = net(img)
        y_att = torch.zeros_like(label) + int(tar)
        with torch.no_grad():
            metric.add(d2l.accuracy(y_hat, y_att), y_att.numel())
    return metric[0] / metric[1]