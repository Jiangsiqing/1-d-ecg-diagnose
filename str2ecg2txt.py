import os
import pandas as pd
import numpy as np
cishu=0
path_train=r'/home/jiangsiqing/yunxindian/ecg_pytorch/data/hf_round1_train/train'
path_test=r'/home/jiangsiqing/yunxindian/ecg_pytorch/data/hf_round1_testA/testA'
for k in os.listdir(path_test):
    path=os.path.join(path_test,k)
    a=pd.read_table(path).values
    def str2int(strlist):
        b=[]
        for i in strlist:
            for j in i.split(' '):
                b.append(float(j))
        b.insert(2,b[1]-b[0])
        b.append(-(b[1]+b[0])/2)
        b.append((b[0]-b[1])/2)
        b.append((b[1]-b[0])/2)
        return b
    g=[]
    for i in a:
        g.append(str2int(i))
    g=np.array(g)
    name=['I','II','III','V1','V2','V3','V4','V5','V6','aVR','aVL','aVF']
    csvfile=pd.DataFrame(data=g,columns=name)
    k=str(k.split('.')[0])+'.csv'
    csvfile.to_csv(r'/data2/jiangsiqing/ecg/test/%s'%k,index=False)
    cishu+=1
    print('finish%i'%cishu)

