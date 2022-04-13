import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
# path=r'/home/jiangsiqing/yunxindian/ecg_all'
# # path1=r'D:\ecg_data\S_lic'
# # ecg_path=os.path.join(path,os.listdir(path))
# # ecpat=os.path.join(r'D:\ecg_data\S_lic',os.listdir(r'D:\ecg_data\S_lic'))
# lll=0
# for j in os.listdir(path):
#     ecg_path = os.path.join(path, j)
#     # print(ecg_path)
#     # print(int(j.split('.')[0]))
#     df=pd.read_csv(ecg_path).values[:,0]
#     n=[]
#     for i in range(len(df)-1):
#         if df[i]==0 and df[i+1]==0 :
#             n.append(i)
#     if n==[]:
#         n=[400]
#     df=df[:n[0]]
#     x=np.arange(len(df))
#     # plt.figure(figsize=(512,512))
#     # plt.plot(x,df)
#     # plt.savefig(r'C:\Users\realdoctor\Desktop\5120.png')
#     coeffs = pywt.wavedec(df, 'db8')
#     for i in range(1,len(coeffs)):
#         coeffs[i]=pywt.threshold(coeffs[i], 0.05*max(coeffs[i]))
#     datarec = pywt.waverec(coeffs, 'db8')
#     # plt.figure()
#     # plt.subplot(2, 1, 1)
#     # plt.xticks([]),plt.yticks([])
#     # plt.plot(np.arange(len(df)),df)
#     # plt.subplot(2, 1, 2)
#     plt.xticks([]),plt.yticks([])
#     plt.plot(np.arange(len(datarec)), datarec)
#     plt.savefig(r'/home/jiangsiqing/yunxindian/ecg_all_img/%i.png'%(int(j.split('.')[0])))
#     lll+=1
#     print('finish %i'%int(j.split('.')[0]),'完成%i幅图'%lll)
#     plt.close()
# # plt.tight_layout()
# # plt.show()


path=r'/data2/jiangsiqing/ecg/hf_round1_train/train'
path1=r'/data2/jiangsiqing/ecg/test'
lll=0
for j in os.listdir(path1):
    b=os.path.join(path1,j)
    df=pd.read_csv(b).values[:,0]
    coeffs = pywt.wavedec(df, 'db3')
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], 0.05 * max(coeffs[i]))
    datarec = pywt.waverec(coeffs, 'db3')
    # plt.figure(figsize=(4,3))
    plt.xticks([]), plt.yticks([])
    plt.plot(np.arange(len(datarec)), datarec)
    # plt.savefig(r'/data2/jiangsiqing/ecgimg/hf_round1_train/train/%i.png'%(int(j.split('.')[0])))
    plt.savefig(r'/data2/jiangsiqing/ecgimg/test/%i.png' % (int(j.split('.')[0])))
    lll+=1
    print('finishi %i'%lll)
    plt.close()
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.xticks([]),plt.yticks([])
# plt.plot(np.arange(len(df)),df)
# plt.subplot(2, 1, 2)
# plt.xticks([]), plt.yticks([])
# plt.plot(np.arange(len(datarec)), datarec)
# plt.show()