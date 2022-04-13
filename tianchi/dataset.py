  # !/usr/bin/env Python
  # coding=utf-8
import pywt, os, copy
import torch
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from scipy import signal
import cv2 as cv
import random
from pyts.image import GramianAngularField

def resample(sig, target_point_num=None):

    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return sig

def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise

def verflip(sig):

    return sig[::-1, :]

def shift(sig, interval=20):

    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset
    return sig


def transform(sig, train=False):

    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.5: sig = verflip(sig)
        if np.random.randn() > 0.5: sig = shift(sig)

    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig


class ECGDataset(Dataset):
    def __init__(self, data_path, train=True):
        super(ECGDataset, self).__init__()
        dd = torch.load(config.train_data)
        self.train = train
        if self.train:
            self.data1d = dd['train1d']
            self.dataimg=dd['trainimg']
        else:
            self.data1d =dd['val1d']
            self.dataimg = dd['valimg']
        self.idx2name = dd['idx2name']
        self.file2idx = dd['file2idx']
        self.png2tar=dd['png2tar']
        self.csv2tar=dd['csv2tar']
        self.wc = 1. / np.log(dd['wc'])
        

    def __getitem__(self, index):
        fid = self.data1d[index]
        img=self.dataimg[index]
        gasf = GramianAngularField(image_size=500, method='summation')
        file_path = os.path.join(config.train_dir, fid)
        img_path=os.path.join(config.train_img,img)
        x = pd.read_csv(file_path).values
        rand=random.choice(range(4500))
        gaf1 = gasf.fit_transform(x[rand:rand+500,0].reshape(1,500))
        gaf1 = torch.from_numpy(gaf1)
        gaf2 = gasf.fit_transform(x[rand:rand+500,1].reshape(1,500))
        gaf2 = torch.from_numpy(gaf2)
        gaf3 = gasf.fit_transform(x[rand:rand+500,2].reshape(1,500))
        gaf3 = torch.from_numpy(gaf3)
        gaf4 = gasf.fit_transform(x[rand:rand+500,3].reshape(1,500))
        gaf4 = torch.from_numpy(gaf4)
        gaf5 = gasf.fit_transform(x[rand:rand+500,4].reshape(1,500))
        gaf5 = torch.from_numpy(gaf5)
        gaf6 = gasf.fit_transform(x[rand:rand+500,5].reshape(1,500))
        gaf6 = torch.from_numpy(gaf6)
        gaf7 = gasf.fit_transform(x[rand:rand+500,6].reshape(1,500))
        gaf7 = torch.from_numpy(gaf7)
        gaf8 = gasf.fit_transform(x[rand:rand+500,7].reshape(1,500))
        gaf8 = torch.from_numpy(gaf8)
        gaf9 = gasf.fit_transform(x[rand:rand+500,8].reshape(1,500))
        gaf9 = torch.from_numpy(gaf9)
        gaf10 = gasf.fit_transform(x[rand:rand+500,9].reshape(1,500))
        gaf10 = torch.from_numpy(gaf10)
        gaf11 = gasf.fit_transform(x[rand:rand+500,10].reshape(1,500))
        gaf11 = torch.from_numpy(gaf11)
        gaf12 = gasf.fit_transform(x[rand:rand+500,11].reshape(1,500))
        gaf12 = torch.from_numpy(gaf12)
        gaf=torch.cat([gaf1,gaf2,gaf3,gaf4,gaf5,gaf6,gaf7,gaf8,gaf9,gaf10,gaf11,gaf12],dim=0)
        a = cv.imread(img_path)[60:420,82:562]
        img_gray=torch.from_numpy(a)
        x=torch.tensor(x)
        # x = transform(df, self.train)
        target = np.zeros(config.num_classes)
        target[self.csv2tar[fid]] = 1
        target = torch.tensor(target, dtype=torch.float32)
        # print(type(x),type(target))
        return x, img_gray,gaf,target

    def __len__(self):
        return len(self.data1d)


if __name__ == '__main__':
    d = ECGDataset(data_path=config.train_data,train=False)
    # print(len(d))
    from torch.utils.data import DataLoader
    train_dataset = ECGDataset(data_path=config.train_data, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=6)
    for x,img,gaf,tar in train_dataloader:
        print(x.shape,img.shape,gaf.shape,tar.shape)
    # for inp,tar in train_dataloader:
        # print(inp.shape,tar.shape)
    # print(train_dataloader[0])