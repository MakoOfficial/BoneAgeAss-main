import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from PIL import Image
from PIL import Image, ImageOps, ImageEnhance, __version__
from torch.utils.data import WeightedRandomSampler
import os
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt


# 将指定文件夹下所有png格式的图片转化为灰度图并存放在GrayPicture当中
def all_file_path(*, Input_root, Output_root):
    for root, dirs, files in os.walk(Input_root):
        for file in files:
            if ".png" in file:
                file_path = os.path.join(root, file)
                __picture2gray(Input_root=Input_root, filePathName=file_path, outputRoot=Output_root)


def __picture2gray(*, Input_root, filePathName, outputRoot):
    # 转化成灰度图
    gray_transforms = transforms.Compose([transforms.Grayscale(1)])
    img = gray_transforms(Image.open(filePathName))
    # 得到存储灰度图的文件路径
    outputFilePath = filePathName.replace(Input_root, outputRoot)
    # 提前检测存储文件对应的目录存不存在；如果不存在就创建
    outputFileRoot = outputFilePath[:outputFilePath.rfind('/')]
    if not os.path.exists(outputFileRoot):
        os.makedirs(outputFileRoot)
    # 存储灰度图
    img.save(outputFilePath)


def train_test_division(data, age_label, sex_label, proportion):
    size = int(proportion * data.shape[0])
    train_data, train_age_label, train_sex_label = data[:size, :, :, :], age_label[:size, ], sex_label[:size, ]
    test_data, test_age_label, test_sex_label = data[size:, :, :, :], age_label[size:, ], sex_label[size:, ]

    return train_data, train_age_label, train_sex_label, test_data, test_age_label, test_sex_label


def read_picture(*, rootPath, reshape_size, record_num):
    loader = transforms.Compose([transforms.ToTensor()])
    resize = transforms.Resize([reshape_size, reshape_size])
    output = torch.zeros((1, reshape_size, reshape_size))
    for _, _, files in os.walk(rootPath):
        files = sorted(files)
        for file in files:
            if ".png" in file:
                img = loader(resize(Image.open(os.path.join(rootPath, file))))
                output = torch.cat([output, img], dim=0)
            if record_num + 1 == output.shape[0]:
                break
    return output[1:, :, :].unsqueeze(1)


def read_label(*, filePath, record_num):
    # 3_20 谭天改64行
    data = np.array(pd.read_csv(filePath).sort_values(by="id", ascending=True))[:record_num, 1:]
    # data = np.array(pd.read_csv(filePath))[:record_num, 1:]
    age_label_temp = data[:, :1]
    sex_label_temp = data[:, 1:]

    age_label = torch.zeros((record_num, 230))
    sex_label = torch.zeros((record_num, 2))
    for i in range(record_num):
        age_label[i][age_label_temp[i][0]] = 1
        sex_label[i][sex_label_temp[i][0]] = 1

    return age_label, sex_label


def RunModuleTrainBatch(*, module, train_data, train_sex, real_label, batch_size=32, lr, EPOCH, modelOutputPath='./module.pth'):
    dataset = Data.TensorDataset(train_data, train_sex, real_label)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(module.parameters(), lr=lr)
    
    WeightedSampler = WeightedRandomSampler(getLabelWeight(real_label),train_data.shape[0],True)

    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler = WeightedSampler
    )

    for epoch in tqdm(range(EPOCH)):
        pred_list = []  #
        grand_age = []  #
        curEpochLoss = torch.tensor(0.0)
        for step, (batch_x, batch_sex, batch_y) in enumerate(loader):
            optimizer.zero_grad()
            pred = module(batch_x.cuda(), batch_sex.cuda()) #
            loss = criterion(pred, batch_y.cuda())  #
            curEpochLoss = curEpochLoss + loss
            loss.backward()
            optimizer.step()
            pred = pred.cpu()   #
            pred = pred.argmax(axis=1)  #
            grand_true = batch_y.argmax(axis=1) #
            pred_list.extend(pred)  #
            grand_age.extend(grand_true)   #
        print('\n测试输出\n预测值 实际值')    #
        for pred0, grand_true in zip(pred_list, grand_age):     #
            print(f'{round(pred0.item(), 2)},   {round(grand_true.item(), 2)}')     #
        print('------------------------------------------')  #
        print('当前轮次为:' + str(epoch) + '，MAE损失为:' + str(curEpochLoss.item()))
        # LossRecord.append(curEpochLoss)

    # LossRecord = torch.tensor(LossRecord, device="cpu")
    # plt.plot(LossRecord)
    # plt.show()

    if modelOutputPath != '':
        print('模型已经存储')
        torch.save(module, modelOutputPath)

    return module


def testModule(*, module, test_data, test_sex, real_label):
    loss1 = nn.MSELoss()
    loss2 = nn.L1Loss()
    res_loss1 = 0
    res_loss2 = 0

    dataset = Data.TensorDataset(test_data, test_sex, real_label)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=False,
    )

    for step, (batch_x, batch_sex, batch_y) in enumerate(loader):
        pred_label = torch.argmax(module(batch_x.cuda(), batch_sex.cuda()), dim=1).float()
        real_label = torch.argmax(batch_y.cuda(), dim=1).float()
        res_loss1 += loss1(pred_label, real_label)
        res_loss2 += loss2(pred_label, real_label)
        print(pred_label, real_label)

    print('MSE误差为:' + str(res_loss1))
    print('MAE误差为:' + str(res_loss2))

#    解决样本分配，得到再分配的权重。
def getLabelWeight(label):
    label = torch.argmax(label, dim=1)
    record = torch.zeros(230)
    for i in range(label.shape[0]):
        record[label[i]] += 1

    record = label.shape[0] / record
    record = torch.where(torch.isinf(record), torch.full_like(record, 0), record)
    res = torch.zeros((label.shape[0], 1))
    for i in range(label.shape[0]):
        res[i] = record[label[i]]

    return res.squeeze(1)