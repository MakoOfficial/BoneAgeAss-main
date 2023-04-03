import pandas as pd
from PIL import Image, ImageOps
import os
import numpy as np
import shutil
from torchvision import transforms
import cv2
from albumentations.augmentations.transforms import Lambda, RandomBrightnessContrast
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate, HorizontalFlip
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.crops.transforms import RandomResizedCrop
from albumentations import Compose
import collections
import math
import csv


"""从12611个训练集中获取1000张图片，并对少于10张的标签增强到10张，最终能够获得2000+图片"""


aug_dataset = Compose([
    # 随机大小裁剪，512为调整后的图片大小，（0.5,1.0）为scale剪切的占比范围，概率p为0.5
    RandomResizedCrop(512, 512, (0.5, 1.0), p=0.5),
    # ShiftScaleRotate操作：仿射变换，shift为平移，scale为缩放比率，rotate为旋转角度范围，border_mode用于外推法的标记，value即为padding_value，前者用到的，p为概率
    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=0.0,
                     p=0.8),
    # 水平翻转
    HorizontalFlip(p=0.5),
    # 概率调整图片的对比度
    RandomBrightnessContrast(p=0.8, contrast_limit=(-0.3, 0.2)),
    # 将图片转化为tensor类型
    ToTensorV2()
])

def copyfile(fname, target_dir):
    """将文件复制到指定文件夹"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(fname, target_dir)


def read_a_image(data_dir, fname, image_size=512):
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
    img = np.array(ImageOps.expand(img, padding).convert("RGB"))
    return img

# d2l里的获取标签
def read_csv_labels(fname):
    """读取标签，返回字典格式`"""
    with open(fname, 'r') as f:
        # Skip the file header line (column name) 跳过文件头
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    # return dict(((id, boneage) for id, boneage, male in tokens))
    return dict((id, [boneage, male]) for id, boneage, male in tokens)

def creatSmallDataset(data_dir, labels, target_size, DF):
    """创建新的数据集"""
    train_list = [['id', 'boneage', 'male']]
    age_list = list(np.array(list(labels.values()))[:, 0])
    class_count = collections.Counter(age_list)
    myBatch = {}
    lengthOfDataset = len(labels)
    for key in class_count.keys():
        myBatch[key] = math.ceil(target_size * class_count[key] / lengthOfDataset)
    label_count = {}
    for idx, row in DF.iterrows():
        # 获取当前文件路径
        filename = os.path.join(data_dir, 'boneage-training-dataset', f"{row['id']}.png")
        # 获取该文件的标签
        label = str(row['boneage'])
        if label not in label_count or label_count[label] < myBatch[label]:
            copyfile(filename, os.path.join(data_dir, 'small-dataset', label))
            train_list.append([row['id'], row['boneage'], row['male']])
            # 将该标签加入记录， 若count中没有则默认返回默认值0
            label_count[label] = label_count.get(label, 0) + 1
    with open(os.path.join(data_dir, 'small-dataset.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in train_list:
            writer.writerow(row)

def reorg_SmallDataset(data_dir, labels, target_size, DF):
    """创建新的数据集"""
    train_list = [['id', 'boneage', 'male']]
    valid_list = [['id', 'boneage', 'male']]
    age_list = list(np.array(list(labels.values()))[:, 0])
    class_count = collections.Counter(age_list)
    myBatch = {}
    lengthOfDataset = len(labels)
    for key in class_count.keys():
        myBatch[key] = math.ceil(target_size * class_count[key] / lengthOfDataset)
    label_count = {}
    for idx, row in DF.iterrows():
        # 获取当前文件路径
        filename = os.path.join(data_dir, 'small-dataset', f"{row['id']}.png")
        # 获取该文件的标签
        label = str(row['boneage'])
        if label not in label_count or label_count[label] < myBatch[label]:
            copyfile(filename, os.path.join(data_dir, 'testDataset', 'valid-dataset'))
            valid_list.append([row['id'], row['boneage'], row['male']])
            # 将该标签加入记录， 若count中没有则默认返回默认值0
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(filename, os.path.join(data_dir, 'testDataset', 'train-dataset'))
            train_list.append([row['id'], row['boneage'], row['male']])
    with open(os.path.join(data_dir, 'testDataset', 'train-dataset.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in train_list:
            writer.writerow(row)
    with open(os.path.join(data_dir, 'testDataset', 'valid-dataset.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in valid_list:
            writer.writerow(row)

def save_one_augpic(root, data_dir, fname, label, aug_idx, aug):
    img = read_a_image(data_dir=data_dir, fname=fname)
    img = aug(image=img)['image']
    img = transforms.ToPILImage()(img)
    id = fname.split('.')[0]+aug_idx
    age = label[0]
    male = label[1]
    row = [id, age, male]
    save_path = f'{data_dir}/{id}.png'
    img.save(save_path)
    with open(os.path.join(root, 'small-dataset.csv'), 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)


def aug_all(root_path, dataset_file, aug, aug_size, label_list):
    """对文件夹下的所有图片进行增广"""
    input_root = os.path.join(root_path, dataset_file)
    for root, dirs, files in os.walk(input_root):
        dirs_size = len(dirs)
        files_size = len(files)
        if dirs_size == 0 and files_size < aug_size:
            need_size = aug_size - files_size
            files = sorted(files)
            for i in range(need_size):
                new_id = '0'+str(i)
                idx = i % files_size
                fname = files[idx]
                label = label_list[fname.split('.')[0]]
                save_one_augpic(root=root_path, data_dir=root, fname=fname, label=label, aug_idx=new_id, aug=aug)


def move_all_files(dir_path, des_file):
    """将所有的文件移动到指定文件夹"""
    if os.path.exists(dir_path):
        #   初始文件夹路径存在，获取文件夹中的所有文件
        path_list = os.listdir(dir_path)
        # 遍历所有文件
        for each_path in path_list:
            # 如果当前文件是文件夹，对当前文件夹重新执行move_all_files方法
            if ".png" in each_path:
                # 当前文件路径
                src_file = dir_path + "/" + each_path
                # 复制
                shutil.move(src_file, des_file)
            else:
                src = dir_path + "/" + each_path
                move_all_files(src, des_file)
    else:
        print("指定路径不存在")

def reorg_aug_data(data_dir, target_size, aug_size):
    labels = read_csv_labels(os.path.join(data_dir, 'boneage-training-dataset.csv'))
    df = pd.read_csv(os.path.join(data_dir, 'boneage-training-dataset.csv'))
    creatSmallDataset(data_dir, labels, target_size, df)
    aug_all(data_dir, 'small-dataset', aug_dataset, aug_size, labels)
    move_all_files(os.path.join(data_dir, 'small-dataset'), os.path.join(data_dir, 'small-dataset'))
    new_labels = read_csv_labels(os.path.join(data_dir, 'small-dataset.csv'))
    new_df = pd.read_csv(os.path.join(data_dir, 'small-dataset.csv'))
    reorg_SmallDataset(data_dir, new_labels, 100, new_df)


if __name__ == '__main__':
    data_dir = '../data/archive/'
    labels = read_csv_labels(os.path.join(data_dir, 'boneage-training-dataset.csv'))
    target_size = 1000
    aug_size = 10
    # move_all_files(os.path.join(data_dir, 'small-dataset'), os.path.join(data_dir, 'small-dataset'))
    reorg_aug_data(data_dir=data_dir, target_size=target_size, aug_size=aug_size)