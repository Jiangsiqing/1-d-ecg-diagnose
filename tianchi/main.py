  # !/usr/bin/env Python
  # coding=utf-8
import torch, time, os, shutil
import models, utils
import numpy as np
import pandas as pd
from tensorboard_logger import Logger
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import ECGDataset
from config import config
import matplotlib.pyplot as plt
import tqdm
import modelss
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from modelss import resnet34,ViT,alexnet
# from models import resnet34


os.environ["CUDA_VISIBLE_DEVICES"] = '9'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(41)
torch.cuda.manual_seed(41)

def save_best_model(state, is_best, model_save_dir,epoch):
    current_w = os.path.join(model_save_dir, 'current_w.pkl')
    best_w = os.path.join(model_save_dir, 'best_w.pkl')
    torch.save(state, current_w)
    if is_best:
        shutil.copyfile(current_w, best_w)
        print('****||Updata the best model in epoch%i!||****'%epoch)

class resnetlstm_vit(nn.Module):
    def __init__(self):
        super(resnetlstm_vit, self).__init__()
        self.reslstm=resnet34()
        # self.vit=ViT(
        # image_height = 360,
        # image_width = 480,
        # patch_height = 60,
        # patch_width = 80,
        # num_classes = 55,
        # dim = 1024,
        # depth = 6,
        # heads = 16,
        # mlp_dim = 2048,
        # dropout = 0.1,
        # emb_dropout = 0.1)
        # self.alex=alexnet()
        # self.linear=nn.Linear()
    def forward(self,x1,x2,x3):
        reslstm=self.reslstm(x1)
        # vit=self.vit(x2)
        # vit=vit.view(-1,100,55)
        # gaf=self.resnet34_2d(x3)
        # gaf=self.alex(x3)
        # gaf=gaf.view(-1,100,55)
        # print(reslstm.shape,vit.shape)
        # x=torch.cat((reslstm,vit),0)
        # x=torch.cat([reslstm,vit,gaf],dim=2)
        # print(x.shape)
        x=reslstm
        return x

# def save_ckpt(state, is_best, model_save_dir):
#     current_w = os.path.join(model_save_dir, config.current_w)
#     best_w = os.path.join(model_save_dir, config.best_w)
#     torch.save(state, current_w)
#     if is_best:
#         shutil.copyfile(current_w, best_w)
#         print('save the best models!!!')

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

def get_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return f1_score(y_true, y_pre)

def train_epoch(model, epoch,optimizer,criterion, train_dataloader, show_interval=10):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    loss_step,Acc,Recall,Pre,F1=0,0,0,0,0
    tq = tqdm.tqdm(total=len(train_dataloader))
    tq.set_description('epoch {}'.format(epoch))
    for i,(inputs, img,gaf,target) in enumerate(train_dataloader):
        if i<269:
            tq.update(1)
            inputs=inputs.type(torch.FloatTensor)
            inputs=inputs.view(-1,5000,12)
            inputs = inputs.to(device)
            img=img.type(torch.FloatTensor)
            img = img.view(-1,3,360,480)
            img = img.to(device)
            gaf = gaf.type(torch.FloatTensor)
            gaf = gaf.to(device)
            target = target.type(torch.FloatTensor)
            # target=target.view(-1,100,55)
            target=target.view(100,55)
            target = target.to(device)
            #target=target.view(-1,100,55)
            #print(target.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            output = model(inputs,img,gaf)
            #print(output.shape)
            # n=torch.sigmoid(output)
            # y_pre = n.view(-1).cpu().detach().numpy() > 0.5
            # print(target,target.shape,output.shape,y_pre,output)
            loss = criterion(output, target)
            # loss=1-get_f1(target,output)
            loss.backward()
            optimizer.step()
            loss_meter += loss.item()
            it_count += 1
            loss_step+=1
            # F1 = utils.calc_f1(target, output)
            # f1 = F1 / loss_step
            Acc += get_acc(target, output)
            acc = Acc / loss_step
            Recall += get_recall(target, output)
            recall = Recall / loss_step
            Pre += get_pre(target, output)
            pre = Pre / loss_step
            F1+=2/(1/pre+1/recall)
            f1=F1/it_count
            if it_count != 0 and it_count % show_interval == 0:
                print("train_step:%d || train_loss:%.3e ||" %(it_count, loss.item()),
                      'f1:', f1, '||', 'acc:', acc, '||', 'recall:', recall, '||', 'pre:', pre)

    tq.close()
    return loss_meter / it_count, f1


def val_epoch(model,epoch, criterion, val_dataloader, threshold=0.5,show_interval=2):
    model.eval()
    f1_meter, loss_meter, it_count = 0, 0, 0
    loss_step,Acc,Recall,Pre,F1=0,0,0,0,0
    tq = tqdm.tqdm(total=len(val_dataloader))
    tq.set_description('epoch {}'.format(epoch))
    with torch.no_grad():
        for i, (inputs, img, gaf, target) in enumerate(val_dataloader):
            tq.update(1)
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.view(-1, 5000, 12)
            inputs = inputs.to(device)
            img = img.type(torch.FloatTensor)
            img = img.view(-1, 3, 360, 480)
            img = img.to(device)
            gaf = gaf.type(torch.FloatTensor)
            gaf = gaf.to(device)
            target = target.type(torch.FloatTensor)
            target = target.view(100, 55)
            target = target.to(device)
            output = model(inputs,img,gaf)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1
            loss_step+=1
            Acc += get_acc(target, output)
            acc = Acc / loss_step
            Recall += get_recall(target, output)
            recall = Recall / loss_step
            Pre += get_pre(target, output)
            pre = Pre / loss_step
            # F1+=2/(1/pre+1/recall)
            F1+=get_f1(target,output)
            f1=F1/it_count
            if it_count != 0 and it_count % show_interval == 0:
                print("train_step:%d || train_loss:%.3e ||" %(it_count, loss.item()),
                      'f1:', f1, '||', 'acc:', acc, '||', 'recall:', recall, '||', 'pre:', pre)
    tq.close()
    return loss_meter / it_count, f1,acc,recall,pre


def train(args):
    # model
    model = resnetlstm_vit()
    if args.ckpt and not args.resume:
        state = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(state['state_dict'])
        print('train with pretrained weight val_f1', state['f1'])
    model = model.to(device)
    # data
    train_dataset = ECGDataset(data_path=config.train_data, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=6)
    #val_dataset = ECGDataset(data_path=config.train_data, train=False)
    #val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4)
    # print("train_datasize", len(train_dataset), "val_datasize", len(val_dataset))
    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr,weight_decay=0.0002)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-4,
                                                           threshold_mode='rel', cooldown=5, min_lr=1e-8)
    w = torch.tensor(train_dataset.wc, dtype=torch.float).to(device)
    criterion = utils.WeightedMultilabel(w)
    # criterion=nn.BCEWithLogitsLoss()
    model_save_dir = '%s/%s_%s' % (config.ckpt, 'resnetlstm_vit+',time.strftime("%Y%m%d%H%M"))
    if args.ex: model_save_dir += args.ex
    best_f1 = -1
    lr = config.lr
    start_epoch = 1
    stage = 1

    if args.resume:
        if os.path.exists(args.ckpt):  
            model_save_dir = args.ckpt
            current_w = torch.load(os.path.join(args.ckpt, config.current_w))
            best_w = torch.load(os.path.join(model_save_dir, config.best_w))
            best_f1 = best_w['loss']
            start_epoch = current_w['epoch'] + 1
            lr = current_w['lr']
            stage = current_w['stage']
            model.load_state_dict(current_w['state_dict'])

            if start_epoch - 1 in config.stage_epoch:
                stage += 1
                lr /= config.lr_decay
                utils.adjust_learning_rate(optimizer, lr)
                model.load_state_dict(best_w['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(start_epoch - 1))
    logger = Logger(logdir=model_save_dir, flush_secs=2)
    for epoch in range(start_epoch, config.max_epoch + 1):
        since = time.time()
        train_loss, train_f1 = train_epoch(model, epoch,optimizer, criterion, train_dataloader, show_interval=10)
        #val_loss, val_f1 = val_epoch(model, epoch,criterion, val_dataloader,show_interval=50)
        print('train loss:',train_loss,'||','train f1:',train_f1)
        logger.log_value('train_loss', train_loss, step=epoch)
        logger.log_value('train_f1', train_f1, step=epoch)
        state = model.state_dict()
        save_best_model(state, best_f1 < train_f1, model_save_dir,epoch=epoch)
        best_f1 = max(best_f1, train_f1)
        if epoch in config.stage_epoch:
            stage += 1
            lr /= config.lr_decay
            best_w = os.path.join(model_save_dir, 'best_w.pkl')
            model.load_state_dict(torch.load(best_w))
            print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            utils.adjust_learning_rate(optimizer, lr)





def val(args):
    model = resnetlstm_vit()
    model.load_state_dict(torch.load('/home/jiangsiqing/yunxindian/tianchi/ckpt/resnetlstm_vit+_202110141146/best_w.pkl'))
    model=model.to(device)
    val_dataset=ECGDataset(data_path=config.train_data, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=True, num_workers=2)
    val_loss_mean,f1,acc,recall,pre= val_epoch(model,criterion=nn.BCEWithLogitsLoss(), epoch=0, val_dataloader=val_dataloader)
    print( 'val_f1:',f1, ' || ','val_acc:',acc, ' || ','val_recall:',recall, ' || ','val_pre:',pre, ' || ', 'loss_mean:', val_loss_mean)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>", help="train or infer")
    parser.add_argument("--ckpt", type=str, help="the path of model weight file")
    parser.add_argument("--ex", type=str, help="experience name")
    parser.add_argument("--resume", action='store_true', default=False)
    args = parser.parse_args()
    if (args.command == "train"):
        train(args)
    if (args.command == "val"):
        val(args)