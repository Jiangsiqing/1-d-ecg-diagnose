import os
import numpy as np

np.random.seed(2021)

def split_train_val(class_path,split_rate):
    train_data=[]
    val_data=[]
    for i in os.listdir(class_path):
        data_list=os.listdir(os.path.join(class_path,i))#A_lic path
        train_split=data_list[30:int(len(data_list)*split_rate)+30]#A_lic_train
        for j in train_split:
            train_data.append(j)
        val_data_small=set(data_list).difference(set(train_data))
        for i in val_data_small:
            val_data.append(i)

    print('len of train_data:',len(train_data),'len of val_data:',len(val_data))
    return train_data,val_data



def k_split(class_path,k,split_rate):
    train_class_k,train_k=[],[]
    val_class_k,val_k=[],[]
    for i in os.listdir(class_path):
        data_list=os.listdir(os.path.join(class_path,i))
        num_val=int(len(data_list)*(1-split_rate))
        for m in range(k):
            val_split=data_list[m*num_val:(m+1)*num_val]
            train_split=list(set(data_list).difference(set(val_split)))
            val_class_k.append(val_split)
            train_class_k.append(train_split)

    classes=len(os.listdir(class_path))
    a=[]
    x=[]
    
    for m in range(k):
        for i in range(classes):
            a.append(m + k * i)

        for n in a:
            for j in val_class_k[n]:
                x.append(j)
        val_k.append(x)
        x = []
        a = []
    print('len of val_k_1:',len(val_k[1]))

    # val_class_k=[[5],[4],[3],[2],[1],[10],[9],[8],[7],[6],[15],[14],[13],[12],[11],[20],[19],[18],[17],[16],[25],[24],[23],[22],[21],[30],[29],[28],[27],[26],[35],[34],[33],[32],[31]]
    for m in range(k):
        for i in range(classes):
            a.append(m + k * i)
        for n in a:
            for j in train_class_k[n]:
                x.append(j)
        train_k.append(x)
        x = []
        a = []
    print('len of train_k_1:',len(train_k[1]))
    # print(train_k[0][:100],val_k[0][:100])
    return train_k,val_k

# if __name__ == '__main__':
#     k_split(class_path='/home/jiangsiqing/yunxindian/ecg_segg',k=5,split_rate=0.8)



def count_label(class_path):
    classes=len(os.listdir(class_path))
    a=classes*[0]
    for i in os.listdir(class_path):
        class_list=os.listdir(class_path)
        # [j for j, x in enumerate(class_list) if x == i]
        a[class_list.index(i)]=len(os.listdir(os.path.join(class_path,i)))
    print('count_label:',a)
    return a


# if __name__ == '__main__':
#     count_label(r'D:\ecg_segg')

def csvname_to_target(class_path=r'D:\ecg_segg'):
    file2tar=dict()
    class_list = os.listdir(class_path)
    a=[0]*len(class_list)
    for i in class_list:
        for j in os.listdir(os.path.join(class_path,i)):
            a[class_list.index(i)]=1
            file2tar[j]=a
            a=[0]*len(class_list)
    print('len of file2tar:',len(file2tar))
    return file2tar

# if __name__ == '__main__':
#     csvname_to_target()

def csvname_to_index(class_path=r'D:\ecg_segg'):
    file2idx=dict()
    class_list = os.listdir(class_path)
    for i in class_list:
        for j in os.listdir(os.path.join(class_path,i)):
            file2idx[j]=[class_list.index(i)]
    print('len of file2idx:',len(file2idx))
    return file2idx


def csvname_to_filename(class_path):
    csv2name=dict()
    class_list=os.listdir(class_path)
    for i in class_list:
        for j in os.listdir(os.path.join(class_path,i)):
            csv2name[j]=i
    print(csv2name,'len of csv2name:',len(csv2name)) 
    return csv2name       
# if __name__ == '__main__':
#     csvname_to_index()

def save_pth(class_path,root,k_fold=False):
    import torch
    count = count_label(class_path)
    csv2tar = csvname_to_target(class_path)
    csv2idx = csvname_to_index(class_path)
    #idx2name={[0]:'N_lic',[1]:'F_lic',[2]:'R_lic',[3]:'A_lic',[4]:'L_lic',[5]:'V_lic',[6]:'Q_lic'}
    csv2name=csvname_to_filename(class_path)

    if k_fold:
        train, val = k_split(class_path, k=5, split_rate=0.8)
    else:
        train, val = split_train_val(class_path,split_rate=0.9)

    train_val_count_csv2index_csv2tar = {'train': train, 'val': val, 'count_label': count, 'csv2tar': csv2tar,
                                         'csv2idx': csv2idx,'csv2name':csv2name}
    save_path = os.path.join(root, 'MIT_BIH train.pth')
    torch.save(train_val_count_csv2index_csv2tar, save_path)
    
def train_txt():
    import torch
    import os
    a=torch.load(r'/home/jiangsiqing/yunxindian/MIT_BIH train.pth')
    text=open(r'/home/jiangsiqing/yunxindian/train.txt','w')
    for i in a['train']:
        path_list=os.path.join(r'/home/jiangsiqing/yunxindian/ecg_all',i)
        text.write(path_list+'\n')
    text.close()
    print('GET train.txt!!!')

if __name__ == '__main__':
    save_pth(class_path=r'/home/jiangsiqing/yunxindian/ecg_segg',root=r'/home/jiangsiqing/yunxindian')
# if __name__ == '__main__':
#     train_txt()

