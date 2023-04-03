import pandas as pd
from PIL import Image, ImageOps
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import shutil
from torchvision import transforms
import cv2
from albumentations.augmentations.transforms import Lambda, RandomBrightnessContrast
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate, HorizontalFlip
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.crops.transforms import RandomResizedCrop
from albumentations import Compose
import torch.nn as nn
import torch.utils.data as Data
import torch.utils.data.dataset as Dataset
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
import mymodel
from d2l import torch as d2l
import collections
import matplotlib.pyplot as  plt
import print_animator

"""本文档主要是解决训练过程中的一些所用到的函数的集合
函数列表：
获取神经网络：get_net
标准化数组:standardization
标准化每个通道:sample_normalize
训练集的数据增广:training_compose
"""


def get_net(attention_size=256, feature_channels=2048, output_channels=1024, isEnsemble=True):
    """获取神经网络，attention_size是指注意力机制Q，K矩阵的长度（default=256）， feature_channels为MMAC输出的通道数（default=2048），output_channels为GA模块输出的注意力图通道数（default=1024）
    isEnsemble是指调用的是整体MMANet（default=TRUE），若值为False，则只调用前半部分的ResNet+MMCA"""
    if isEnsemble:
        # MMANet_beforeGA = torch.load('./MMANet_BeforeGA.pth')
        MMANet_beforeGA = torch.load('./module.pth')
        GA = mymodel.GA(attention_size, feature_channels, output_channels, MMANet_beforeGA)
        MMANet = mymodel.MMANet(GA)
    else:
        MMANet = mymodel.MMANet_BeforeGA(32, *mymodel.get_ResNet())
    return MMANet


def standardization(data):
    """标准化数组"""
    mean, std = data.mean(axis=0), data.std(axis=0)
    return mean, std, (data - mean) / std


def sample_normalize(image, **kwargs):
    """标准化每个通道"""
    image = image / 255
    channel = image.shape[2]
    mean, std = image.reshape((-1, channel)).mean(axis=0), image.reshape((-1, channel)).std(axis=0)
    return (image - mean) / (std + 1e-3)


# 随机删除一个图片上的像素，p为执行概率，scale擦除部分占据图片比例的范围，ratio擦除部分的长宽比范围
randomErasing = transforms.RandomErasing(scale=(0.02, 0.08), ratio=(0.5, 2), p=0.8)


def randomErase(image, **kwargs):
    """随机删除一个图片上的像素"""
    return randomErasing(image)


transform_train = Compose([
    # 训练集的数据增广
    # 随机大小裁剪，512为调整后的图片大小，（0.5,1.0）为scale剪切的占比范围，概率p为0.5
    RandomResizedCrop(512, 512, (0.5, 1.0), p=0.5),
    # ShiftScaleRotate操作：仿射变换，shift为平移，scale为缩放比率，rotate为旋转角度范围，border_mode用于外推法的标记，value即为padding_value，前者用到的，p为概率
    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=0.0,
                     p=0.8),
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


transform_valid = Compose([
    # 验证集的数据处理
    Lambda(image=sample_normalize),
    ToTensorV2()
])


# def read_image(data_dir, fname, image_size=512):
#     """读取图片，并统一修改为512x512"""
#     img = Image.open(os.path.join(data_dir, fname))
#     # 开始修改尺寸
#     w, h = img.size
#     long = max(w, h)
#     # 按比例缩放成512
#     w, h = int(w / long * image_size), int(h / long * image_size)
#     # 压缩并插值
#     img = img.resize((w, h), Image.ANTIALIAS)
#     # 然后是给短边扩充，使用ImageOps.expand
#     delta_w, delta_h = image_size - w, image_size - h
#     padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
#     # 转化成np.array，并通过compose输出，输出格式为(3, 512, 512)
#     img = transform_train(image=np.array(ImageOps.expand(img, padding).convert("RGB")))['image']
#     return torch.reshape(img, (1, 3, 512, 512))

def read_image(data_dir, fname, image_size=512):
    """读取图片，并统一修改为512x512"""
    img = Image.open(os.path.join(data_dir, fname))
    # 开始修改尺寸
    w, h = img.size
    long = max(w, h)
    # 按比例缩放成512
    w, h = int(w / long * image_size), int(h / long * image_size)
    # 压缩并插值
    img = img.resize((w, h), Image.ANTIALIAS)
    # 然后是给短边扩充，使用ImageOps.expand
    delta_w, delta_h = image_size - w, image_size - h
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    # 转化成np.array
    return np.array(ImageOps.expand(img, padding).convert("RGB"))

# create 'dataset's subclass,we can read a picture when we need in training trough this way

# class TrainDataset(Dataset):
#     def __init__(self, df, file_path) -> None:
#         def preprocess_df(df):
#             # nomalize boneage data_distribution，对选中的元素减去平均值，然后除以标准差，并将这个重新设置一个叫zscore的列
#             # df['zscore'] = df['boneage'].map(lambda x: (x - boneage_mean)/boneage_div )
#             # change the type of gender, change bool variable to float32，将性别和年龄转化为float32
#             df['male'] = df['male'].astype('float32')
#             df['bonage'] = df['boneage'].astype('float32')
#             return df
#
#         self.df = preprocess_df(df)
#         self.file_path = file_path
#
#     def __getitem__(self, index):
#         row = self.df.iloc[index]
#         num = int(row['id'])
#         return (transform_train(image=read_image(self.file_path, f"{num}.png"))['image'], Tensor([row['male']])), row['boneage']
#
#     def __len__(self):
#         return len(self.df)

def read_all_image(*, rootPath):
    """读取文件夹下的所有图片，问题：读的太慢了，900M读半天"""
    output = torch.zeros((1, 3, 512, 512))
    count = 0
    for root, dirs, files in os.walk(rootPath):
        files.sort(key=lambda x: int(x[:-4]))
        for fname in files:
            if ".png" in fname:
                count += 1
                img = read_image(root, fname)
                output = torch.cat([output, img], dim=0)
                print(count)
    return output[1:, :, :, :]


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def copyfile(fname, target_dir):
    """将文件复制到指定文件夹"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(fname, target_dir)


def read_labels(data_dir, fname):
    """读取标签，并将其标准化"""
    data = pd.read_csv(os.path.join(data_dir, fname))
    data = data.sort_values(by="id", ascending=True)
    data['male'] = data['male'].astype('float32')
    data['boneage'] = data['boneage'].astype('float32')
    data = np.array(data)
    Id = data[:, :1]
    age = data[:, 1:2]
    gender = data[:, 2:]
    return torch.tensor(Id), torch.tensor(age), torch.tensor(gender)


def get_weight(labels):
    """得出每个年龄段在整个数据集中的权重，权重公式：总数/该年龄段样本数，为权重采样器服务"""
    record = torch.zeros(230)
    for i in range(labels.shape[0]):
        record[int(labels[i])] += 1

    record = labels.shape[0] / record
    # torch.where(condition, a, b)，按照条件整合两个tensor类型，满足条件选a，不满足则b
    record = torch.where(torch.isinf(record), torch.full_like(record, 0), record)
    res = torch.zeros((labels.shape[0], 1))
    # 将根据record的统计，给每个样本的权重赋值
    for i in range(labels.shape[0]):
        res[i] = record[int(labels[i])]

    return res.squeeze(1)

def sort_by_key(d):
    """按键值的数字大小给字典排序， 输入必须要是字典"""
    # return sorted(d.items(), key=lambda k:k[0].astype('int'))
    return sorted(d.items())

def list_class_counter(input):
    """得到一个列表中所有类别的数量，返回counter类型，key和values都为int"""
    input = list(map(lambda x: int(x), input))
    input = collections.Counter(input)
    return input

def print_data_distribution(output, grand_true):
    output = list_class_counter(output)
    grand_true = list_class_counter(grand_true)
    output = dict(sort_by_key(output))
    grand_true = dict(sort_by_key(grand_true))
    legend = ['output', 'grand true']
    animator = print_animator.Animator(xlabel='month', xlim=[1, 230], legend=legend)
    for valid, true in zip(output.items(), grand_true.items()):
        age_valid, num_valid = valid
        age_true, num_true = true
        animator.add(age_valid, (num_valid, num_true))
    animator.save('data_distribution.png')

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

    return AM1, AM2, AM3, AM4, feature_map, texture, gender_encode, accuracy(pred_list[1:, :], grand_age[1:])

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.

    Defined in :numref:`sec_fashion_mnist`"""
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img, cmap='gray')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    fig.savefig('./attentionMap.png')
    return axes

def apply(img, num_cols, num_rows=5):
    show_images(img, num_rows, num_cols, scale=2)

def print_AM(file_path, AM1, AM2, AM3, AM4):
    plt_pic = []
    for root, dirs, files in os.walk(file_path):
        files.sort(key=lambda x: int(x[:-4]))
        for file in files:
            if ".png" in file:
                img = Image.open(os.path.join(root, file))
                plt_pic.append(img)
    for i in range(AM1.shape[0]):
        img = AM1[i, :, :, :].squeeze()
        img = img.cpu().clone()
        # img = unloader(img)
        plt_pic.append(img)
    for i in range(AM2.shape[0]):
        img = AM2[i, :, :, :].squeeze()
        img = img.cpu().clone()
        # img = unloader(img)
        plt_pic.append(img)
    for i in range(AM3.shape[0]):
        img = AM3[i, :, :, :].squeeze()
        img = img.cpu().clone()
        # img = unloader(img)
        plt_pic.append(img)
    for i in range(AM4.shape[0]):
        img = AM4[i, :, :, :].squeeze()
        img = img.cpu().clone()
        # img = unloader(img)
        plt_pic.append(img)
    apply(img=plt_pic, num_cols=AM1.shape[0])

def L1_penalty(net, alpha):
    """罚函数"""
    loss = 0
    for param in net.MLP.parameters():
        loss += torch.sum(torch.abs(param))

    return alpha * loss


# loss = nn.CrossEntropyLoss(reduction="none")
loss = nn.L1Loss(reduction='sum')


def train_eachRun(net, train_data, train_age, train_gender, test_data, test_age, test_gender, num_epoch, lr, wd, lr_period, lr_decay, batch_size=32,
                isEnsemble=False):
    """训练函数，输入参数：网络，训练数据，训练性别，训练年龄，测试数据，测试性别，测试年龄，批量大小，轮数，学习率，暂退法比率，学习率衰减轮数，学习率衰减比率，模型存储路径"""
    # 标准化age
    # mean, std, zscore = standardization(train_age)
    data_size = len(train_data)
    # dataset = Data.TensorDataset(train_data, train_gender, zscore)
    dataset = Data.TensorDataset(train_data, train_gender, train_age)
    # 定义损失函数
    loss = nn.L1Loss(reduction='sum')
    # 定义Adam优化器
    # trainer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    # 每隔几个周期，学习率就衰减
    # scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    # 设置权重采样器
    WeightedSampler = WeightedRandomSampler(get_weight(train_age), train_data.shape[0], True)
    # 定义迭代器
    train_iter = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=WeightedSampler,
        drop_last=True
    )
    # train_ls, test_ls = [], []
    # 迭代器读取的轮数
    net = net.to(device=try_gpu())
    net.fine_tune()
    # 加载动画
    # animator = print_animator.Animator(xlabel='epoch', ylabel='MAE', xlim=[1, num_epoch], legend=['train', 'valid'])
    for epoch in tqdm(range(num_epoch)):
        # batch_num = 0
        # this_epoch_loss = []
        epoch_loss = torch.tensor(0.0)
        # pred_list = []
        # grand_age = []
        # batch_mae = []
        for idx, (batch_data, batch_gender, batch_age) in enumerate(train_iter):
            # 优化器梯度清0
            # this_batch_size = len(batch_data)
            trainer.zero_grad()
            # 获得当前数据的预测值
            batch_data = batch_data.type(torch.FloatTensor)
            batch_gender = batch_gender.type(torch.FloatTensor)
            _, _, _, pred = net(batch_data.cuda(), batch_gender.cuda())
            pred = pred.squeeze().float()
            # 放入损失函数得出误差值， 注意输入的size，使用squeeze压缩一下
            batch_age = batch_age.squeeze().float().cuda()
            l = loss(pred, batch_age)
            # 加入罚函数
            # total_loss = l + L1_penalty(net, 1e-5)
            # total_loss = l
            epoch_loss = epoch_loss + l
            # 反向传播
            # print(f'L:{l.item()}, total loss:{total_loss.item()}')
            # total_loss.backward()
            l.backward()
            trainer.step()
            # pred = pred.cpu()
            # batch_age = batch_age.cpu()
            # this_batch_loss = loss(pred.detach()*std + mean, batch_age.detach()*std + mean).item()
            # this_epoch_loss.append(this_batch_loss)
            # batch_mae.append(this_batch_loss/this_batch_size)
            
        # 一个批量训练完后将损失总值放入列表末尾
        # bacth_plt_x = list(range(1, len(this_epoch_loss)+1))
        # bacth_plt_y = batch_mae
        # fig, (ax1, ax2) = plt.subplots(2, 1)
        # fig = plt.figure(figsize=(7, 5))
        # ax1 = fig.add_subplot(211)
        # plt.subplots(211)
        # ax[0][0].plot(bacth_plt_x, bacth_plt_y, 'g-', label=u'batch loss')
        # ax1.plot(bacth_plt_x, bacth_plt_y)
        # ax1.set_xlabel('batch sequence')
        # ax1.set_ylabel('MAE')
        # ax1.set_title('batch loss')
        # plt.legend()
        # plt.xlabel(u'batch sequence')
        # plt.ylabel(u'MAE')
        # plt.title('batch loss in one epoch')
        # plt.draw()
        # plt.pause(6)
        # plt.close(fig)
        # plt.show()
        # train_ls.append(sum(this_epoch_loss) / data_size)

        # print(f'epoch: {epoch+1}')
        # print(f'train loss :{this_epoch_loss / data_size}')
        # print('预测值 实际值')
        # print('训练结果')
        # for pred, grand_true in zip(pred_list[0], grand_age[0]):
        #     print(f'{round(pred.item(), 2)},   {round(grand_true.item(), 2)}')
        # 开始验证
        # if test_age is not None:
        #     test_loss_batch = valid_fn(net=net, valid_data=test_data, valid_age=test_age, valid_gender=test_gender, batch_size=batch_size)
        #     test_ls.append(test_loss_batch)
            # print(f'valid loss: {test_loss_batch}')
        # ax2 = fig.add_subplot(211)
        # plt.subplots(212)
        # plt.subplots(2, 1, 1)
        # epoch_x = list(range(0, epoch + 1))
        # epoch_y1 = train_ls
        # epoch_y2 = test_ls
        # ax[1][0].plot(epoch_x, epoch_y1, 'b-', label=u'training loss')
        # ax2.plot(epoch_x, epoch_y1)
        # plt.legend()
        # ax[1][0].plot(epoch_x, epoch_y2, 'r-', label=u'valid loss')
        # ax2.plot(epoch_x, epoch_y2)
        # ax2.set_xlabel('epoch')
        # ax2.set_ylabel('MAE')
        # ax2.set_title('Compare loss between training and valid')
        # plt.legend()
        # plt.xlabel(u'epoch')
        # plt.ylabel(u'MAE')
        # plt.title('Compare loss between training and valid')
        # animator.add(epoch+1, (train_ls[-1], test_ls[-1]))
        # plt.draw()
        # plt_save_path = './batch_loss/batch_loss.png'
        # plt.savefig(plt_save_path)
        # img = Image.open(plt_save_path)
        # plt.imshow(img)
        # plt.draw()
        # plt.show()
        # plt.close(fig)
        # plt_save_path = './batch_loss/'+str(epoch)+'_epoch_loss.png'
        # fig.savefig(plt_save_path)
        # print(f'epoch:{epoch + 1}，训练MAE{float(train_ls[-1]):f}, '
        #       f'验证MAE{float(test_ls[-1]):f}')
        print(f'epoch:{epoch + 1}，训练MAE{float(epoch_loss.item()/data_size)}')
        # scheduler.step()
        # torch.cuda.empty_cache()
    # d2l.plot(list(range(1, num_epoch + 1)), [train_ls, test_ls],
    #          xlabel='epoch', ylabel='MAE', xlim=[1, num_epoch],
    #          legend=['train', 'valid'])
    # animator.show()
    # 是否存储当前模型
    # if isEnsemble:
    #     print('MMANet已存储')
    #     torch.save(net, './MMANet.pth')
    # else:
    #     print('MMANet_BeforeGA已存储')
    #     torch.save(net, './MMANet_BeforeGA_test.pth')

    # return train_ls, test_ls

def valid_fn(*, net, val_loader, device):
    """验证函数：输入参数：网络，验证数据，验证性别，验证标签
    输出：返回MAE损失"""
    mae_loss = torch.tensor([0], dtype=torch.float32)
    val_total_size = torch.tensor([0], dtype=torch.float32)    # 记录验证集的大小
    accuracy_num = torch.tensor([0], dtype=torch.float32)
    net.fine_tune(False)
    # MAE = nn.L1Loss(reduction='sum')
    MAE = nn.L1Loss(reduction='none')
    this
    with torch.no_grad():
        pred_list = []  # 获取所有的预测值
        grand_age = []  # 获得所有的真实值
        for batch_idx, data in enumerate(val_loader):
            # val_total_size += len(data[1])
            image, gender = data[0]
            image, gender = image.type(torch.FloatTensor).to(device), gender.type(torch.FloatTensor).to(device)
            label = data[1].type(torch.FloatTensor).to(device)
            val_total_size += label.numel()
            #   net内求出的是normalize后的数据，这里应该是是其还原，而不是直接net（）
            _, _, _, y_pred = net(image, gender)
            # y_pred = y_pred * boneage_div + boneage_mean
            # y_pred = net(image, gender)
            # y_pred = y_pred.squeeze()
            accuracy_num += d2l.accuracy(y_pred, label)
            y_pred = y_pred.argmax(axis=1)  # 由于输出的是长度为230的向量，所以最大值所对应的坐标就是预测值
            batch_loss = MAE(y_pred, label).item()
            mae_loss += batch_loss
            pred_list.extend(y_pred.detach().cpu())
            grand_age.extend(label)
        print('\n测试输出\n预测值    实际值')
        for pred0, grand_true in zip(pred_list, grand_age):
            print(f'{int(pred0.item())},   {int(grand_true.item())}')
        print('------------------------------------------')
        val_mae = mae_loss / val_total_size
        print(f'val loss is {val_mae}, val accuracy is {accuracy_num / val_total_size}')
    return val_mae

# def valid_fn(*, net, valid_data, valid_age, valid_gender, batch_size):
#     """验证函数：输入参数：网络，验证数据，验证性别，验证标签
#     输出：返回MAE损失"""
#     net.fine_tune(False)
#     MSE = nn.MSELoss(reduction='sum')
#     MAE = nn.L1Loss(reduction='sum')
#     data_size = len(valid_data)
#     MSE_loss = 0
#     MAE_loss = 0
#
#     mean, std, zscore = standardization(valid_age)
#     dataset = Data.TensorDataset(valid_data, zscore, valid_gender)
#     valid_iter = Data.DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         drop_last=True
#     )
#     net = net.to(device=try_gpu())
#     with torch.no_grad():
#         pred_list = []
#         grand_age = []
#         for idx, (batch_data, batch_age, batch_gender) in enumerate(valid_iter):
#             batch_data = batch_data.type(torch.FloatTensor)
#             batch_gender = batch_gender.type(torch.FloatTensor)
#             _, _, _, pred = net(batch_data.cuda(), batch_gender.cuda())
#             pred = pred.squeeze().float().cpu()
#             batch_age = batch_age.squeeze().float().cpu()
#             MSE_loss += MSE(pred*std + mean, batch_age*std + mean).item()
#             MAE_loss += MAE(pred*std + mean, batch_age*std + mean).item()
#             # pred_list.append(pred*std + mean)
#             # grand_age.append(batch_age*std + mean)
#             pred_list.extend(pred*std + mean)
#             grand_age.extend(batch_age*std + mean)
#         print('\n测试输出\n预测值 实际值')
#         # for i in range(len(pred_list)):
#         #     for pred0, grand_true in zip(pred_list[i], grand_age[i]):
#         #         print(f'{round(pred0.item(), 2)},   {round(grand_true.item(), 2)}')
#         for pred0, grand_true in zip(pred_list, grand_age):
#             print(f'{round(pred0.item(), 2)},   {round(grand_true.item(), 2)}')
#         print('------------------------------------------')
#         print_data_distribution(pred_list, grand_age)
#     MSE_loss = MSE_loss / data_size
#     MAE_loss = MAE_loss / data_size
#     return MAE_loss

def get_k_fole_data(k, i, data, age, gender):
    """k折获取验证集和训练集"""
    # 确保k合法
    assert k > 1
    fold_size = data.shape[0] // k
    data_train, age_train, gender_train, data_valid, gender_valid, age_valid = None, None, None, None, None, None
    for j in range(k):
        if j==(k-1):
            idx = slice(j * fold_size, len(data))
        else:
            idx = slice(j * fold_size, (j + 1) * fold_size)
        data_part, gender_part, age_part = data[idx, :], gender[idx, :], age[idx, :]
        if i == j:
            data_valid, gender_valid, age_valid = data_part, gender_part, age_part
        elif data_train == None:
            data_train, gender_train, age_train = data_part, gender_part, age_part
        else:
            data_train = torch.cat([data_train, data_part], dim=0)
            gender_train = torch.cat([gender_train, gender_part], dim=0)
            age_train = torch.cat([age_train, age_part], dim=0)
    return data_train, age_train, gender_train, data_valid, age_valid, gender_valid

def k_fold(k, train_data, train_age, train_gender, num_epochs, learning_rate, weight_decay, batch_size, lr_period, lr_decay, isEnsemble=False):
    """k折交叉验证"""
    train_l_sum, valid_l_sum = 0, 0
    # animator = d2l.Animator(xlabel='epoch', ylabel='MAE', xlim=[1, num_epochs], legend=['train', 'valid'])
    for i in range(k):
        data = get_k_fole_data(k, i, train_data, train_age, train_gender)
        net = get_net(isEnsemble=isEnsemble)
        train_ls, valid_ls = train_eachRun(net, *data, num_epoch=num_epochs, lr=learning_rate, wd=weight_decay, lr_period=lr_period, lr_decay=lr_decay, batch_size=batch_size)
        train_ls = torch.tensor(train_ls, device='cpu')
        valid_ls = torch.tensor(valid_ls, device='cpu')
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        # if i == 0:
        #     d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
        #              xlabel='epoch', ylabel='loss', xlim=[1, num_epochs],
        #              legend=['train', 'valid'], yscale='log')
        # animator.add(i+1, (train_ls[-1], valid_ls[-1]))
        print(f'折{i + 1}，训练loss: {float(train_ls[-1]):f}, '
              f'验证loss: {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

def get_mean(list):
    """求一个list的均值，用于对K折交叉验证时，求出K折的损失均值"""
    return sum(list)/len(list)

def two_label(net, train_data, train_age, train_gender, num_epochs, learning_rate, batch_size, lr_period, lr_decay):
    data_size = len(train_data)
    dataset = Data.TensorDataset(train_data, train_gender, train_age)
    # 定义损失函数
    # loss = nn.L1Loss(reduction='sum')
    criterion = nn.BCEWithLogitsLoss()
    # 定义Adam优化器
    trainer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    train_iter = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size
    )
    for epoch in tqdm(range(num_epochs)):
        pred_list = []  #
        grand_age = []  #
        curEpochLoss = torch.tensor(0.0)
        for idx, (batch_data, batch_gender, batch_age) in enumerate(train_iter):
            # 优化器梯度清0
            trainer.zero_grad()
            # 获得当前数据的预测值
            batch_data = batch_data.type(torch.FloatTensor)
            batch_gender = batch_gender.type(torch.FloatTensor)
            _, _, _, pred = net(batch_data.cuda(), batch_gender.cuda())
            # 放入损失函数得出误差值， 注意输入的size，使用squeeze压缩一下
            batch_age = batch_age.cuda()
            l = criterion(pred, batch_age)
            # 反向传播
            l.backward()
            trainer.step()
            curEpochLoss = curEpochLoss + l.detach().item()
            pred = pred.cpu()   #
            pred = pred.argmax(axis=1)  #
            grand_true = batch_age.argmax(axis=1) #
            pred_list.extend(pred)  #
            grand_age.extend(grand_true)   #
        print('\n测试输出\n预测值 实际值')    #
        for pred0, grand_true in zip(pred_list, grand_age):     #
            print(f'{round(pred0.item(), 2)},   {round(grand_true.item(), 2)}')     #
        print('------------------------------------------')  #
        print('当前轮次为:' + str(epoch) + '，MAE损失为:' + str(curEpochLoss.item()))
        torch.cuda.empty_cache()
        scheduler.step()

if __name__ == '__main__':
    # num_epochs, learning_rate, weight_decay = 10, 2e-4, 5e-4
    # lr_period, lr_decay = 10, 0.5
    # x = torch.rand((16, 3, 512, 512))
    # gender = torch.ones((16, 1))
    # age = torch.ones((16, 1))
    MMANet = get_net(isEnsemble=True)
    MMANet = MMANet.to(device=try_gpu())
    # print(sum(p.numel() for p in MMANet.parameters()))
    params = list(MMANet.MLP.parameters())
    print(params)
    # loss = valid_fn(net=MMANet, valid_data=x, valid_age=age, valid_gender=gender, batch_size=4)
    # print(loss)
    # data_dir = './dataset/'
    # output = read_all_image(rootPath='./dataset/small-training-dataset')
    # # print(output.shape)
    # Id, age, gender = read_labels(data_dir, 'small-training-dataset.csv')
    # print(age.mean())
    # zscore = standardization(age)
    # print(zscore.mean())
    # data_train, age_train, gender_train, data_valid, age_valid, gender_valid = get_k_fole_data(10, 0, output, age, gender)


