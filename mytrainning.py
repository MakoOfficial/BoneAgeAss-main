from PIL import Image, ImageOps
import os
import numpy as np
import pandas as pd
import shutil
import torch
from torch import Tensor
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import Dataset
# from model import BAA_New, get_My_resnet50
import myKit
import warnings
import time
from d2l import torch as d2l
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import csv
warnings.filterwarnings("ignore")

""""还未进行重构的训练函数"""

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def copyfile(fname, target_dir):
    """将文件复制到指定文件夹"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(fname, target_dir)

from albumentations.augmentations.transforms import Lambda, RandomBrightnessContrast
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate, HorizontalFlip
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.crops.transforms import RandomResizedCrop
from albumentations import Compose

import cv2
from torchvision import transforms


# 随机删除一个图片上的像素，p为执行概率，scale擦除部分占据图片比例的范围，ratio擦除部分的长宽比范围
randomErasing = transforms.RandomErasing(scale=(0.02, 0.08), ratio=(0.5, 2), p=0.8)

def randomErase(image, **kwargs):
    """"""
    return randomErasing(image)


# 标准化每个通道
def sample_normalize(image, **kwargs):
    image = image/255
    channel = image.shape[2]
    mean, std = image.reshape((-1, channel)).mean(axis = 0), image.reshape((-1, channel)).std(axis = 0)
    return (image - mean)/(std + 1e-3)

# 训练集的图像增广

transform_train = Compose([
    # 随机大小裁剪，512为调整后的图片大小，（0.5,1.0）为scale剪切的占比范围，概率p为0.5
    RandomResizedCrop(512, 512, (0.5, 1.0), p=0.5),
    # ShiftScaleRotate操作：仿射变换，shift为平移，scale为缩放比率，rotate为旋转角度范围，border_mode用于外推法的标记，value即为padding_value，前者用到的，p为概率
    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=0.0, p=0.8),
    # 水平翻转
    HorizontalFlip(p=0.5),
    # 概率调整图片的对比度
    RandomBrightnessContrast(p=0.8, contrast_limit=(-0.3, 0.2)),
    # 标准化
    Lambda(image=sample_normalize),
    # 将图片转化为tensor类型
    ToTensorV2(),
    # 做随机擦除
    Lambda(image=randomErase)
])

# 验证集的数据处理
transform_valid = Compose([
    Lambda(image=sample_normalize),
    ToTensorV2()
])

def read_iamge(data_dir, fname, image_size=512):
    img = Image.open(os.path.join(data_dir, fname))
    # 开始修改尺寸
    w, h = img.size
    long = max(w, h)
    # 按比例缩放成512
    w, h = int(w/long*image_size), int(h/long*image_size)
    # 压缩并插值
    img = img.resize((w, h), Image.ANTIALIAS)
    # 然后是给短边扩充，使用ImageOps.expand
    delta_w, delta_h = image_size - w, image_size - h
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    # 转化成np.array再返回
    return np.array(ImageOps.expand(img, padding).convert("RGB"))

# 罚函数
def L1_penalty(net, alpha):
    loss = 0
    for param in net.output.parameters():
        loss += torch.sum(torch.abs(param))

    return alpha*loss

# 定义损失函数
# loss = nn.CrossEntropyLoss(reduction="none")
loss = nn.L1Loss(reduction='sum')


# 重写getitem函数，这样可以将这个类类似于DF
class BAATrainDataset(Dataset):
    def __init__(self, df, file_path) -> None:
        def preprocess_df(df):
            #nomalize boneage data_distribution，对选中的元素减去平均值，然后除以标准差，并将这个重新设置一个叫zscore的列
            # df['zscore'] = df['boneage'].map(lambda x: (x - boneage_mean)/boneage_div )
            #change the type of gender, change bool variable to float32，将性别和年龄转化为float32
            df['male'] = df['male'].astype('float32')
            df['bonage'] = df['boneage'].astype('float32')
            return df

        self.df = preprocess_df(df)
        self.file_path = file_path

    def __getitem__(self, index):
        row = self.df.iloc[index]
        num = int(row['id'])
        return (transform_train(image=read_iamge(self.file_path, f"{num}.png"))['image'], Tensor([row['male']])), row['boneage']

    def __len__(self):
        return len(self.df)

class BAAValDataset(Dataset):

    def __init__(self, df, file_path) -> None:
        def preprocess_df(df):
            #change the type of gender, change bool variable to float32
            df['male'] = df['male'].astype('float32')
            df['bonage'] = df['boneage'].astype('float32')
            return df
        self.df = preprocess_df(df)
        self.file_path = file_path

    def __getitem__(self, index):
        row = self.df.iloc[index]
        num = int(row['id'])
        return (transform_valid(image=read_iamge(self.file_path, f"{num}.png"))['image'], Tensor([row['male']])), row['boneage']

    def __len__(self):
        return len(self.df)

def create_data_loader(train_df, val_df, train_root, val_root):
    return BAATrainDataset(train_df, train_root), BAAValDataset(val_df, val_root)

criterion = nn.CrossEntropyLoss(reduction='none')

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))
def evaluate_fn(data_iter, net, devices):
    net.eval()
    pred_list = torch.zeros((1, 230))
    grand_age = torch.zeros((1,))
    with torch.no_grad():
        for batch_idx, data in enumerate(data_iter):
            image, gender = data[0]
            image, gender = image.type(torch.FloatTensor).to(devices), gender.type(torch.FloatTensor).to(devices)
            label = data[1].type(torch.FloatTensor).to(devices)
            AM1, AM2, AM3, AM4, feature_map, texture, gender_encode, y_pred = net(image, gender)
            y_pred = y_pred.cpu()
            label = label.cpu()
            pred_list = torch.cat([pred_list, y_pred], dim=0)
            grand_age = torch.cat([grand_age, label], dim=0)
    return feature_map, texture, gender_encode, accuracy(pred_list[1:, :], grand_age[1:])

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))

def map_fn(flags):
    record = [['epoch', 'training loss', 'val loss', 'acc']]
    with open('./RECORD.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in record:
            writer.writerow(row)
    device = try_gpu()
    # mymodel = BAA_New(32, *get_My_resnet50())
    # mymodel = mymodel.to(device)
    mymodel = myKit.get_net(isEnsemble=False)
    mymodel = mymodel.to(device)
    # 数据读取
    # Creates dataloaders, which load data in batches
    # Note: test loader is not shuffled or sampled
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=flags['batch_size'],
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=flags['batch_size'],
        shuffle=False)


    ## Network, optimizer, and loss function creation
    mymodel = mymodel.train()

    global best_loss
    best_loss = float('inf')
    #   loss_fn =  nn.MSELoss(reduction = 'sum')
    loss_fn = nn.L1Loss(reduction='sum')
    lr = flags['lr']

    wd = 0

    optimizer = torch.optim.Adam(mymodel.parameters(), lr=lr, weight_decay=wd)
    #   optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay = wd)
    # 每过10轮，学习率降低一半
    scheduler = StepLR(optimizer, step_size=2, gamma=0.8)

    ## Trains

    train_start = time.time()
    for epoch in range(flags['num_epochs']):
        print(epoch)
        this_record = []
        global training_loss
        training_loss = torch.tensor([0], dtype=torch.float32)
        global total_size
        total_size = torch.tensor([0], dtype=torch.float32)

        global mae_loss
        mae_loss = torch.tensor([0], dtype=torch.float32)
        global val_total_size
        val_total_size = torch.tensor([0], dtype=torch.float32)

        # xm.rendezvous("initialization")

        start_time = time.time()
        # 在不同的设备上运行该模型

        #   打开微调
        mymodel.fine_tune()
        #   enumerate（），为括号中序列构建索引
        for batch_idx, data in enumerate(train_loader):
            # #put data to GPU
            image, gender = data[0]
            image, gender = image.type(torch.FloatTensor).to(device), gender.type(torch.FloatTensor).to(device)

            batch_size = len(data[1])
            label = data[1].to(device)

            # zero the parameter gradients，是参数梯度归0
            optimizer.zero_grad()
            # forward
            _, _, _, _, _, _, _, y_pred = mymodel(image, gender)
            # y_pred = y_pred.squeeze()

            # print(y_pred, label)，求损失
            # loss = loss_fn(y_pred, label)
            loss = criterion(y_pred, label.long()).sum()
            # backward,calculate gradients，反馈计算梯度
            # total_loss = loss + L1_penalty(net, 1e-5)
            loss.backward()
            # backward,update parameter，更新参数
            optimizer.step()

            batch_loss = loss.item()

            training_loss += batch_loss
            total_size += batch_size
            print('this batch loss:', batch_loss / batch_size)

        ## Evaluation
        # Sets net to eval and no grad context
        mymodel.fine_tune(False)
        pred_list = torch.zeros((1, 230))
        grand_age = torch.zeros((1,))
        with torch.no_grad():
            # pred_list = []
            # grand_age = []
            for batch_idx, data in enumerate(val_loader):
                val_total_size += len(data[1])

                image, gender = data[0]
                image, gender = image.type(torch.FloatTensor).to(device), gender.type(torch.FloatTensor).to(device)

                label = data[1].type(torch.FloatTensor).to(device)

                #   net内求出的是normalize后的数据，这里应该是是其还原，而不是直接net（）
                _, _, _, _, _, _, _, y_pred = mymodel(image, gender)
                # y_pred = y_pred * boneage_div + boneage_mean
                y_pred_loss = y_pred.argmax(axis=1)
                # y_pred = net(image, gender)
                # y_pred = y_pred.squeeze()

                batch_loss = F.l1_loss(y_pred_loss, label, reduction='sum').item()
                # print(batch_loss/len(data[1]))
                mae_loss += batch_loss
                # pred_list.extend(y_pred.detach().cpu())
                # grand_age.extend(label)
                y_pred = y_pred.cpu()
                label = label.cpu()
                pred_list = torch.cat([pred_list, y_pred], dim=0)
                grand_age = torch.cat([grand_age, label], dim=0)

        #   反向传播更新参数
            # print('\n测试输出\n预测值 实际值')
            # for i in range(len(pred_list)):
            #     for pred0, grand_true in zip(pred_list[i], grand_age[i]):
            #         print(f'{round(pred0.item(), 2)},   {round(grand_true.item(), 2)}')
            # for pred0, grand_true in zip(pred_list, grand_age):
            #     print(f'{round(pred0.item(), 2)},   {round(grand_true.item(), 2)}')
            # print('------------------------------------------')
        accuracy_num = accuracy(pred_list[1:, :], grand_age[1:])
        scheduler.step()

        train_loss, val_mae, accuracy1= training_loss / total_size, mae_loss / val_total_size, accuracy_num / val_total_size
        this_record.append([epoch, round(train_loss.item(), 2), round(val_mae.item(), 2), round(100*accuracy1.item(), 2)])
        with open('./RECORD.csv', 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in this_record:
                writer.writerow(row)
        print(
            f'training loss is {round(train_loss.item(), 2)}, val loss is {round(val_mae.item(), 2)}, acuuracy is {round(100*accuracy1.item(), 2)} time : {round((time.time() - start_time), 2)}, lr:{optimizer.param_groups[0]["lr"]}')
    torch.save(mymodel, './new_model.pth')

if __name__ == '__main__':

    flags = {}
    flags['lr'] = 1e-4
    flags['batch_size'] = 8
    flags['num_epochs'] = 50

    train_df = pd.read_csv('../data/archive/small-dataset.csv')
    val_df = pd.read_csv('../data/archive/valid-dataset.csv')
    boneage_mean = train_df['boneage'].mean()
    boneage_div = train_df['boneage'].std()
    train_set, val_set = create_data_loader(train_df, val_df, '../data/archive/small-dataset',
                                            '../data/archive/valid-dataset')
    torch.set_default_tensor_type('torch.FloatTensor')
    map_fn(flags=flags)
