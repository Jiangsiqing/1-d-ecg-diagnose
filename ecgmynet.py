import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import models
from dataset import ECGDATA
import shutil
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
lr=0.0005

def loss_fuc(train_dataset):
    w = torch.tensor(train_dataset.count_label, dtype=torch.float).to(device)
    criterion=nn.BCEWithLogitsLoss()
    return criterion

def l2_regulation(model,l2_alf):
    l2_loss=0.01
    for param in model.parameters():
        l2_loss+=torch.sum(torch.abs(param))
    return l2_alf*l2_loss

def opt_fuc(model,lr):
    opt=optim.Adam(model.parameters(),lr)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min', factor=0.5, patience=10, threshold=1e-4,threshold_mode='rel', cooldown=5, min_lr=1e-8)
    return opt,scheduler

def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return f1_score(y_true, y_pre)



def get_acc(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return accuracy_score(y_true, y_pre)



def get_recall(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return recall_score(y_true, y_pre)



def get_pre(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return precision_score(y_true, y_pre)

def train_epoch(model,epoch,train_dataloader,train_dataset):
    step,LOSS,loss_step,loss_mean=0,0,0,0
    loss_list=[]
    tq=tqdm.tqdm(total=len(train_dataloader))
    tq.set_description('epoch{}'.format(epoch))
    # conf_zero = np.zeros((5,5),dtype=np.float32)
    a_r_p_all=0
    F1,Acc,Pre,Recall=0,0,0,0
    for i,(input,targets) in enumerate(train_dataloader):
        # torch.utils.data.TensorDataset(input.float(), targets.float())
        tq.update(1)
        if i<491:
            input=input.type(torch.FloatTensor)
            targets=targets.type(torch.FloatTensor)
            targets=targets.view(200,5)
            input=input.to(device)
            targets=targets.to(device)
            opt,scheduler=opt_fuc(model,lr)[0],opt_fuc(model,lr)[1]
            criterion=loss_fuc(train_dataset)
            output=model(input)
            #print(output.shape,input.shape,targets.shape)

            # conf_matrix=confusion_matrix(output,targets).numpy()
            # conf_zero+=conf_matrix
            # if epoch>30:
            #     print(conf_zero)
            #if epoch>3:
            #   print(output,targets)
            loss=criterion(output,targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_step+=1
            LOSS+=loss
            loss_mean=LOSS/(loss_step+0.1)
            # targets=targets.cpu().detach().numpy()
            # output=output.cpu().detach().numpy()
            F1+=calc_f1(targets,output)
            f1=F1/loss_step
            Acc+=get_acc(targets,output)
            acc=Acc/loss_step+0.01
            Recall+=get_recall(targets,output)
            recall=Recall/loss_step+0.01
            Pre+=get_pre(targets,output)
            pre=Pre/loss_step+0.01
            a_r_p_all+=3/(1/acc+1/recall+1/pre)
            # loss_list.append(float(str(loss).split('(')[1].split(',')[0]))

            if i !=0 and i % 50 == 0 and i<=460:
                print('train_loss(epoch=%i,step=%i):'%(epoch,i),str(loss).split('(')[1].split(',')[0],'||','loss_mean_to step=%i:'%i,str(loss_mean).split('(')[1].split(',')[0],' || ',
                      'f1:',f1,'||','acc:',acc,'||','recall:',recall,'||','pre:',pre)
    # x_list = range(0, len(train_dataloader))
    # heat_map(conf_zero+0.001,epoch)
    # if epoch<=3:
    #     plt_figure(x_list,loss_list,xname='step',yname='loss',title='epoch%i'%epoch,epoch=epoch)
    tq.close()
    return str(loss).split('(')[1].split(',')[0],str(loss_mean).split('(')[1].split(',')[0],a_r_p_all/loss_step

class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.conv=nn.Conv1d(2,16,kernel_size=3,stride=1,padding=1)
        self.bn=nn.BatchNorm1d(16)
        self.bn1=nn.BatchNorm1d(64)
        self.relu=nn.ReLU(True)
        self.conv1=nn.Conv1d(16,64,kernel_size=3,stride=1,padding=1)
        self.out = nn.Linear(6400, 512*5)
        self.out1=nn.Linear(512*5,512)
        self.out2=nn.Linear(512,5)
        self.maxpool=nn.MaxPool1d(kernel_size=3,stride=2,padding=1)
        self.drop=nn.Dropout(0.3)
    def forward(self, x):
        x = x.transpose(1, 2)
        x=self.conv(x)
        x=self.drop(x)
        x=self.bn(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.conv1(x)
        x=self.drop(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.drop(x)
        x=x.view(x.size(0),-1)
        out=self.out(x)
        out=self.drop(out)
        out=self.out1(out)
        out=self.drop(out)
        out=self.out2(out)
        return out

class mynet1(nn.Module):#f1=0.94,arp=0.96
    def __init__(self):
        super(mynet1, self).__init__()
        self.conv=nn.Conv1d(2,16,kernel_size=3,stride=1,padding=1)
        #self.bn=nn.BatchNorm1d(16)
        #self.relu=nn.ReLU(True)
        self.out = nn.Linear(3200, 5)
        self.maxpool=nn.MaxPool1d(kernel_size=3,stride=2,padding=1)
        self.drop=nn.Dropout(0.3)
    def forward(self, x):
        x = x.transpose(1, 2)
        x=self.conv(x)
        #x=self.bn(x)
        #x=self.relu(x)
        x=self.maxpool(x)
        x=self.drop(x)
        x=x.view(x.size(0),-1)
        out=self.out(x)
        return out
        
        
def train():
    model=mynet1()
    model=model.to(device)
    train_dataset = ECGDATA(data_path=r'/home/jiangsiqing/yunxindian/ecg_all', train=True,pth_path=r'/home/jiangsiqing/yunxindian/MIT_BIH train.pth')
    train_dataloader = DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=6)
    #some hyperparam
    start_epoch=1
    last_epoch=100
    # best_model=-1
    for epoch in range(start_epoch,last_epoch+1):
        since=time.localtime(time.time())
        train_loss_step,train_loss_mean,a_r_p=train_epoch(model,epoch,train_dataloader=train_dataloader,train_dataset=train_dataset)
        print('epoch:',epoch,' || ',
              'train_loss_step:',train_loss_step,' || ',
              'loss_mean:',train_loss_mean,'||','a_r_p:',a_r_p)
        print('Cost time for train(epoch=%i):%i h| %i m| %i s|'%(epoch,
                                                    time.localtime(time.time()).tm_hour-since.tm_hour,
                                                    time.localtime(time.time()).tm_min-since.tm_min,
                                                    time.localtime(time.time()).tm_sec-since.tm_sec))

if __name__ == '__main__':
    train()