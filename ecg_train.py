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
from models import resnet34,ViT,resnet34_2d,alexnet


batch_size=16
EPOCH=100
lr=0.001


os.environ["CUDA_VISIBLE_DEVICES"] = '7'
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)

class resnetlstm_vit(nn.Module):
    def __init__(self):
        super(resnetlstm_vit, self).__init__()
        self.reslstm=resnet34()
        self.vit=ViT(
        image_height = 360,
        image_width = 480,
        patch_height = 60,
        patch_width = 80,
        num_classes = 5,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1)
        self.resnet34_2d=resnet34_2d()
        self.alex=alexnet()

    def forward(self,x1,x2,x3):
        reslstm=self.reslstm(x1)
        # vit=self.vit(x2)
        # vit=vit.view(-1,200,5)
        # # gaf=self.resnet34_2d(x3)
        # gaf=self.alex(x3)
        # gaf=gaf.view(-1,200,5)
        # print(reslstm.shape,vit.shape)
        # x=torch.cat((reslstm,vit),0)
        x=0.95*reslstm
        return x

def loss_fuc(train_dataset):
    w = torch.tensor(train_dataset.count_label, dtype=torch.float).to(device)
    criterion=nn.BCEWithLogitsLoss()
    return criterion



def opt_fuc(model,lr):
    opt=optim.Adam(model.parameters(),lr)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min', factor=0.5, patience=10, threshold=1e-4,threshold_mode='rel', cooldown=5, min_lr=1e-8)
    return opt,scheduler



def plt_figure(x,y,xname,yname,title,epoch,train=True):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,6))
    plt.plot(x,y)
    plt.xlabel('%s'%xname)
    plt.ylabel('%s'%yname)
    plt.title('%s'%title)
    if train:
        plt.savefig('/home/jiangsiqing/yunxindian/savefig/train_epoch%i.png'%epoch)
    else:
        plt.savefig('/home/jiangsiqing/yunxindian/savefig/val_epoch%i.png'%epoch)
    # plt.show()



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



def save_bast_model(state, is_best, model_save_dir,epoch):
    current_w = os.path.join(model_save_dir, 'current_w.pkl')
    best_w = os.path.join(model_save_dir, 'best_w.pkl')
    torch.save(state, current_w)
    if is_best:
        shutil.copyfile(current_w, best_w)
        print('****||Updata the best model in epoch%i!||****'%epoch)



def confusion_matrix(output,targets):
    o=torch.max(output, 1)[1]
    t=torch.max(targets, 1)[1]
    conf_zero = torch.zeros(5,5)
    for i,j in zip(o,t):
        conf_zero[i,j]+=1
    return conf_zero



def heat_map(conf_matrix,epoch):
    if epoch>=3:
        plt.figure(figsize=(10,6))
        plt.imshow(torch.sigmoid(torch.from_numpy(np.log(conf_matrix))),
                   interpolation='nearest',
                   cmap='bone',
                   origin='lower')
        plt.colorbar(label='sigmoid-log-(samples)')
        plt.xticks(range(5),['N','S','F','V','Q'])
        plt.yticks(range(5),['N','S','F','V','Q'],rotation=0)
        plt.xlabel('GT')
        plt.ylabel('Model_Predict')
        plt.title('Confusion Matrix of (5classes) ECG-signal')
        plt.savefig('/home/jiangsiqing/yunxindian/savefig/heat_map/hmap_epoch%i.png'%epoch)




# def train_epoch(model,epoch,train_dataloader,train_dataset):
#     step,LOSS,loss_step,loss_mean=0,0,0,0
#     loss_list=[]
#     tq=tqdm.tqdm(total=len(train_dataloader))
#     tq.set_description('epoch{}'.format(epoch))
#     conf_zero = np.zeros((5,5),dtype=np.float32)
#     a_r_p_all=0
#     F1,Acc,Pre,Recall=0,0,0,0
#     for i,(input,img_input,targets) in enumerate(train_dataloader):
#         tq.update(1)
#         if i<491:
#             # torch.utils.data.TensorDataset(input.float(), targets.float())
#             input=input.type(torch.FloatTensor)
#             targets=targets.type(torch.FloatTensor)
#             targets=targets.view(-1,200,5)
#             input=input.to(device)
#             targets=targets.to(device)
#             opt,scheduler=opt_fuc(model,lr)[0],opt_fuc(model,lr)[1]
#             opt.zero_grad()
#             criterion=loss_fuc(train_dataset)
#             output=model(input)
#             #conf_matrix=confusion_matrix(output,targets).numpy()
#             #conf_zero+=conf_matrix
#             if epoch>30:
#                 print(conf_zero)
#             loss=criterion(output,targets)
#             loss.backward()
#             opt.step()
#             loss_step+=1
#             LOSS+=loss
#             loss_mean=LOSS/(loss_step+0.1)
#             # targets=targets.cpu().detach().numpy()
#             # output=output.cpu().detach().numpy()
#             F1+=calc_f1(targets,output)
#             f1=F1/loss_step
#             Acc+=get_acc(targets,output)
#             acc=Acc/loss_step
#             Recall+=get_recall(targets,output)
#             recall=Recall/loss_step
#             Pre+=get_pre(targets,output)
#             pre=Pre/loss_step
#             a_r_p_all+=3/(1/acc+1/recall+1/pre)
#             loss_list.append(float(str(loss).split('(')[1].split(',')[0]))
#
#             if i !=0 and i % 50 == 0:
#                 print('train_loss(epoch=%i,step=%i):'%(epoch,i),str(loss).split('(')[1].split(',')[0],'||','loss_mean_to step=%i:'%i,str(loss_mean).split('(')[1].split(',')[0],' || ',
#                       'f1:',f1,'||','acc:',acc,'||','recall:',recall,'||','pre:',pre)
#         x_list = range(0, len(train_dataloader))
#         #heat_map(conf_zero+0.001,epoch)
#         #if epoch<=3:
#         #    plt_figure(x_list,loss_list,xname='step',yname='loss',title='epoch%i'%epoch,epoch=epoch)
#     tq.close()
#     return str(loss).split('(')[1].split(',')[0],str(loss_mean).split('(')[1].split(',')[0],a_r_p_all/loss_step
#         #return str(loss).split('(')[1].split(',')[0],str(loss_mean).split('(')[1].split(',')[0],a_r_p_all/loss_step


# def val_epoch(model,epoch,val_dataloader,val_dataset):
#     step,LOSS,loss_step,loss_mean=0,0,0,0
#     loss_list=[]
#     tq=tqdm.tqdm(total=len(val_dataloader))
#     tq.set_description('epoch{}'.format(epoch))
#     a_r_p_all=0
#     conf_zero = torch.zeros(5,5)
#     F1,Acc,Pre,Recall=0,0,0,0
#     for i,(input,targets) in enumerate(val_dataloader):
#         tq.update(1)
#         input=input.type(torch.FloatTensor)
#         targets=targets.type(torch.FloatTensor)
#         input=input.to(device)
#         targets=targets.to(device)
#         opt,scheduler=opt_fuc(model,lr)[0],opt_fuc(model,lr)[1]
#         opt.zero_grad()
#
#         output=model(input)
#         conf_matrix=confusion_matrix(output,targets)
#         conf_zero+=conf_matrix
#         print(conf_zero)
#         criterion=loss_fuc(val_dataset)
#         loss=criterion(output,targets)
#         loss.backward()
#         opt.step()
#         loss_step+=1
#         LOSS+=loss
#         loss_mean=LOSS/(loss_step+0.1)
#         F1+=calc_f1(targets,output)
#         f1=F1/loss_step
#         Acc+=get_acc(targets,output)
#         acc=Acc/loss_step
#         Recall+=get_recall(targets,output)
#         recall=Recall/loss_step
#         Pre+=get_pre(targets,output)
#         pre=Pre/loss_step
#         a_r_p_all+=3/(1/acc+1/recall+1/pre)
#         loss_list.append(float(str(loss).split('(')[1].split(',')[0]))
#         # x=range(0,len(train_dataloader),len(train_dataloader))
#         x_list=range(0,len(val_dataloader))
#         if i !=0 and i % 20 == 0:
#             print('val_loss(epoch=%i,step=%i):'%(epoch,i),str(loss).split('(')[1].split(',')[0],' || ','loss_mean_to step=%i:'%i,str(loss_mean).split('(')[1].split(',')[0],' || ',
#                   'f1:',f1,'||','acc:',acc,'||','recall:',recall,'||','pre:',pre)
#
#     plt_figure(x_list,loss_list,xname='step',yname='loss',title='epoch%i'%epoch,epoch=epoch,train=False)
#     heat_map(conf_zero,epoch)
#     tq.close()
#     return str(loss).split('(')[1].split(',')[0],str(loss_mean).split('(')[1].split(',')[0],a_r_p_all/loss_step,f1,acc,recall,pre



# def train(args):
#     model=getattr(models,'resnet34')()
#     model=model.to(device)
#     train_dataset = ECGDATA(data_path=r'/home/jiangsiqing/yunxindian/ecg_all', train=True,pth_path=r'/home/jiangsiqing/yunxindian/MIT_BIH train.pth')
#     train_dataloader = DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=6)
#     #some hyperparam
#     start_epoch=1
#     last_epoch=500
#     best_model=-1
#     for epoch in range(start_epoch,last_epoch+1):
#         since=time.localtime(time.time())
#         train_loss_step,train_loss_mean,a_r_p=train_epoch(model,epoch,train_dataloader=train_dataloader,train_dataset=train_dataset)
#         print('epoch:',epoch,' || ',
#               'train_loss_step:',train_loss_step,' || ',
#               'loss_mean:',train_loss_mean,'||','a_r_p:',a_r_p)
#         print('Cost time for train(epoch=%i):%i h| %i m| %i s|'%(epoch,
#                                                     time.localtime(time.time()).tm_hour-since.tm_hour,
#                                                     time.localtime(time.time()).tm_min-since.tm_min,
#                                                     time.localtime(time.time()).tm_sec-since.tm_sec))
#         state = model.state_dict()
#         save_bast_model(state, best_model < a_r_p, model_save_dir=r'/home/jiangsiqing/yunxindian/best_model',epoch=epoch)
#         best_model=max(best_model,a_r_p)
#
#
#
# def val(args):
#     model=getattr(models,'resnet34')()
#     model.load_state_dict(torch.load('/home/jiangsiqing/yunxindian/best_model/best_w.pkl'))
#     model=model.to(device)
#     val_dataset=ECGDATA(data_path=r'/home/jiangsiqing/yunxindian/ecg_all', train=False,pth_path=r'/home/jiangsiqing/yunxindian/MIT_BIH train.pth')
#     val_dataloader = DataLoader(val_dataset, batch_size=50, shuffle=True, num_workers=2)
#     val_loss_step, val_loss_mean ,a_r_p,f1,acc,recall,pre= val_epoch(model, epoch=0, val_dataloader=val_dataloader,val_dataset=val_dataset)
#     print( 'f1:',f1, ' || ','acc:',acc, ' || ','recall:',recall, ' || ','pre:',pre, ' || ', 'loss_mean:', val_loss_mean,'||','a_r_p:',a_r_p)
#

# def L2Loss(model,alpha):
#     l2_loss = torch.tensor(0.0,requires_grad = True)
#     for name,parma in model.named_parameters():
#         if 'bias' not in name:
#             l2_loss = l2_loss + (0.5*alpha * torch.sum(torch.pow(parma,2)))
#     return l2_loss
#
# def L1Loss(model,beta):
#     l1_loss = torch.tensor(0.0,requires_grad = True)
#     for name,parma in model.named_parameters():
#         if 'bias' not in name:
#             l1_loss = l1_loss + beta * torch.sum(torch.abs(parma))
#     return l1_loss

def train_epoch_img(model, epoch, train_dataloader, train_dataset):
    step, LOSS, loss_step, loss_mean = 0, 0, 0, 0
    loss_list = []
    tq = tqdm.tqdm(total=len(train_dataloader))
    tq.set_description('epoch{}'.format(epoch))
    conf_zero = np.zeros((5, 5), dtype=np.float32)
    a_r_p_all = 0
    F1, Acc, Pre, Recall = 0, 0, 0, 0
    for i, (input, img_input,gaf, targets) in enumerate(train_dataloader):
        tq.update(1)
        if i < 491:
            # torch.utils.data.TensorDataset(input.float(), targets.float())
            input = input.type(torch.FloatTensor)
            img_input=img_input.type(torch.FloatTensor)
            gaf=gaf.type(torch.FloatTensor)
            gaf=gaf.view(-1,1,400,400)
            img_input = img_input.view(-1,3,360,480)
            targets = targets.type(torch.FloatTensor)
            targets = targets.view(-1, 200, 5)
            input = input.to(device)
            img_input=img_input.to(device)
            gaf=gaf.to(device)
            targets = targets.to(device)
            opt, scheduler = opt_fuc(model, lr)[0], opt_fuc(model, lr)[1]
            opt.zero_grad()
            criterion = loss_fuc(train_dataset)
            output = model(input,img_input,gaf)
            # m,b=0,0
            # conf_matrix=confusion_matrix(output,targets).numpy()
            # conf_zero+=conf_matrix
            # if epoch > 30:
            #     print(conf_zero)
            loss =criterion(output, targets)
            # reg_loss=torch.tensor(0.0,requires_grad = True)
            for param in model.parameters():
                paramall=torch.sum((param**2).reshape(-1))
            for param in model.parameters():
                paramall+=torch.sum((param**2).reshape(-1))
            reg_loss=0.5*torch.sum(paramall)
            loss+=0.0001*reg_loss
            loss.backward()
            opt.step()
            loss_step += 1
            LOSS += loss
            loss_mean = LOSS / (loss_step + 0.1)
            # targets=targets.cpu().detach().numpy()
            # output=output.cpu().detach().numpy()
            F1 += calc_f1(targets, output)
            f1 = F1 / loss_step
            Acc += get_acc(targets, output)
            acc = Acc / loss_step
            Recall += get_recall(targets, output)
            recall = Recall / loss_step
            Pre += get_pre(targets, output)
            pre = Pre / loss_step
            a_r_p_all += 3 / (1 / acc + 1 / recall + 1 / pre)
            loss_list.append(float(str(loss).split('(')[1].split(',')[0]))

            if i != 0 and i % 50 == 0:
                print('train_loss(epoch=%i,step=%i):' % (epoch, i), str(loss).split('(')[1].split(',')[0], '||',
                      'loss_mean_to step=%i:' % i, str(loss_mean).split('(')[1].split(',')[0], ' || ',
                      'f1:', f1, '||', 'acc:', acc, '||', 'recall:', recall, '||', 'pre:', pre)
        x_list = range(0, len(train_dataloader))
        # heat_map(conf_zero+0.001,epoch)
        # if epoch<=3:
        #    plt_figure(x_list,loss_list,xname='step',yname='loss',title='epoch%i'%epoch,epoch=epoch)
    tq.close()
    return str(loss).split('(')[1].split(',')[0], str(loss_mean).split('(')[1].split(',')[0], a_r_p_all / loss_step
    # return str(loss).split('(')[1].split(',')[0],str(loss_mean).split('(')[1].split(',')[0],a_r_p_all/loss_step

def train_img(args):
    model=resnetlstm_vit()
    model=model.to(device)
    train_dataset = ECGDATA(data_path=r'/home/jiangsiqing/yunxindian/ecg_all',
                            train=True,
                            pth_path=r'/home/jiangsiqing/yunxindian/MIT_BIH train.pth',
                            img_path=r'/home/jiangsiqing/yunxindian/ecg_all_img')
    train_dataloader = DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=6)
    #some hyperparam
    start_epoch=1
    last_epoch=500
    best_model=-1
    for epoch in range(start_epoch,last_epoch+1):
        since=time.localtime(time.time())
        train_loss_step,train_loss_mean,a_r_p=train_epoch_img(model,epoch,train_dataloader=train_dataloader,train_dataset=train_dataset)
        print('epoch:',epoch,' || ',
              'train_loss_step:',train_loss_step,' || ',
              'loss_mean:',train_loss_mean,'||','a_r_p:',a_r_p)
        print('Cost time for train(epoch=%i):%i h| %i m| %i s|'%(epoch,
                                                    time.localtime(time.time()).tm_hour-since.tm_hour,
                                                    time.localtime(time.time()).tm_min-since.tm_min,
                                                    time.localtime(time.time()).tm_sec-since.tm_sec))
        state = model.state_dict()
        save_bast_model(state, best_model < a_r_p, model_save_dir=r'/home/jiangsiqing/yunxindian/best_model',epoch=epoch)
        best_model=max(best_model,a_r_p)

if __name__=='__main__':
    import argparse
    parse=argparse.ArgumentParser()
    parse.add_argument("command",metavar="<command>",help='train or val')
    args=parse.parse_args()
    # if (args.command == "train"):
    #     train(args)
    # if (args.command == "val"):
    #     val(args)
    if (args.command == "train_img"):
        train_img(args)






