import os, copy
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy import signal

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
        # if np.random.randn() > 0.5: sig = shift(sig)

    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig

# a=torch.load(r'C:\Users\realdoctor\Desktop\MIT_BIH train1.pth')

import pandas as pd
import glob
import cv2 as cv
from pyts.image import GramianAngularField
class ECGDATA(Dataset):
    def __init__(self,
                 train=True,
                 data_path=r'/home/jiangsiqing/yunxindian/ecg_all',
                 pth_path=r'/home/jiangsiqing/yunxindian/MIT_BIH train.pth',
                 img_path=r'/home/jiangsiqing/yunxindian/ecg_all_img'):
        super(ECGDATA, self).__init__()
        data=torch.load(pth_path)
        self.train=train
        self.data=data['train'] if train else data['val']
        self.img2tar=data['png2tar']
        self.csv2img=data['csv2png']
        self.count_label=data['count_label']
        self.csv2tar=data['csv2tar']
        self.csv2idx=data['csv2idx']
        self.csv2name=data['csv2name']
        self.data_path=data_path
        self.img_path=img_path
        self.pnglist=data['trainimg']

    def __getitem__(self, index):
        b=self.data[index]
        p=self.pnglist[index]
        gasf = GramianAngularField(image_size=400, method='summation')
        img_path=os.path.join(self.img_path,p)
        data_path=os.path.join(self.data_path,b)
        a = cv.imread(img_path)
        # img_gray = cv.cvtColor(a, cv.COLOR_RGB2GRAY)
        img_gray = torch.from_numpy(a[60:420, 82:562])
        df=pd.read_csv(data_path).values
        num = []
        for i in df:
            if type(i[0]) == np.float64 or type(i[0]) == np.float16 or type(i[0]) == np.float32:
                num = df
            elif type(i[0])==str:
                try:
                    num.append([float(i[0].split(',')[0]), float(i[0].split(',')[1])])
                except:
                    num.append([float(i[0].split('(')[1].split(',')[0]),float(i[1].split('(')[1].split(',')[0])])
        # x = transform(x, self.train)
        
        csvdata=torch.tensor(num)
        gaf = gasf.fit_transform(csvdata[:,0].reshape(1,400))
        gaf = torch.from_numpy(gaf)
        target=self.csv2tar[b]
        target = torch.tensor(target, dtype=torch.float32)

        return csvdata,img_gray,gaf,target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    d=ECGDATA(train=True)
    from torch.utils.data import DataLoader
    train_dataloader=DataLoader(d,batch_size=200,shuffle=True,num_workers=2)
    for step,(x,y,z,t) in enumerate(train_dataloader):
        if step%2 == 0:
            print(step,x.shape,y.shape,z.shape,t.shape)
