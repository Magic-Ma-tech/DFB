import torch.optim as optim

from torch.utils.data import TensorDataset

from models import *

from util import *
import time
from my_model import *
import random
import numpy as np

random_seed = 0
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

transform_normal = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

])

transform_before = transforms.Compose([
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



def encode_train(dataset_path, l_inf_r):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)


    device = torch.device("cuda:0" if True else "cpu")
    save_path = './checkpoint_my/decode_pretrain_binary' + str(200) + '.pth'
    decode_model = ResNet18_201()
    decode_model = nn.DataParallel(decode_model)
    decode_model.load_state_dict(torch.load(save_path))
    decode_model = decode_model.to(device)
    decode_model.eval()

    train_batch_size = 128
    outter_trainset = My_PoisonFolder(1, root=dataset_path + 'tiny-imagenet-200/train/', \
                                      transform=transform_before)

    frog_trainset = My_PoisonFolder(0, root=dataset_path + 'tiny-imagenet-200/frog/', \
                                    transform=transform_before)

    total_set = concoct_dataset(outter_trainset, frog_trainset)


    outter_loader = torch.utils.data.DataLoader(total_set, batch_size=train_batch_size, shuffle=True, num_workers=16)
    frog_loader = torch.utils.data.DataLoader(frog_trainset, batch_size=train_batch_size, shuffle=True, num_workers=16)

    acc_decode_outter_loader = evaluate_test_acc(decode_model, outter_loader, transform_after)

    acc_decode_frog_loader = evaluate_test_acc(decode_model, frog_loader, transform_after)

    if acc_decode_outter_loader< 0.99:
        print(f"fail load acc_decode_outter_loader is {acc_decode_outter_loader}")
    elif acc_decode_frog_loader < 0.99:
        print(f"fail load acc_decode_frog_loader is {acc_decode_frog_loader}")

    else:
        print("success to load decode")

    encode = StegaStampEncoder()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(encode.parameters(), lr=0.0001)



    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        encode = nn.DataParallel(encode)

    encode = encode.to(device)

    scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)

    time_list = []
    for epoch in range(400):
        since = time.time()
        decode_model.eval()
        encode.train()

        loss_list = []
        att_list = []
        correct, total = 0, 0

        # loop = tqdm(frog_loader, total=len(frog_loader))

        for X, y in frog_loader:
            X, y = X.to(device), y.to(device)

            res = encode(X)

            res = torch.clamp(res, -l_inf_r, l_inf_r)

            finish_img = res + X

            calmp_finish = torch.clamp(finish_img, 0, 1)

            trans_finish = transform_after(calmp_finish)

            y = y.reshape(-1)

            y_att = torch.ones_like(y)


            y_hat = decode_model(trans_finish)

            _, predicted = torch.max(y_hat.data, 1)

            total += y_hat.size(0)

            correct += (predicted == y_att).sum().item()

            acc = correct / total

            optimizer.zero_grad()

            l_true = criterion(y_hat, y_att)

            loss = l_true

            loss.backward()

            loss_list.append(float(l_true.data))
            att_list.append(float(acc))

            optimizer.step()
        scheduler_1.step()
        time_elapsed = time.time() - since
        ave_loss = np.average(np.array(l_true.item()))
        att = np.average(np.array(att_list))



        time_list.append(time_elapsed)
        print(f'epoch {epoch}, ave_loss {ave_loss}, frog_to_all_acc {att}\n'
              f'time {time_list[epoch]:.3f}s')

    save_path = './checkpoint_my/encode_pretrain_vgg_l_inf_r_8_frogtoall_test_0-1' + str(200) + '.pth'

    torch.save(encode.state_dict(), save_path)

    print('save success')


if __name__ == '__main__':
    encode_train('./data/', 8/255)




