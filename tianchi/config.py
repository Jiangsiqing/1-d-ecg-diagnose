import os


class Config:
    # for data_process.py
    #root = r'D:\ECG'
    root1 = r'/data2/jiangsiqing/ecg'
    root2=r'/home/jiangsiqing/yunxindian/tianchi'
    train_dir = os.path.join(root1, 'hf_round1_train/train')#1d train
    train_img=os.path.join(r'/data2/jiangsiqing/ecgimg/hf_round1_train', 'train')#2d img
    test_dir = os.path.join(root1, 'test')
    test_img = os.path.join(r'/data2/jiangsiqing/ecgimg','test')
    train_label = os.path.join(root2, 'hf_round1_label.txt')
    test_label = os.path.join(root2, 'hf_round1_subA.txt')
    arrythmia = os.path.join(root2, 'hf_round1_arrythmia.txt')
    train_data = os.path.join(root2, 'train.pth')


    # for train

    model_name = 'resnet34'

    stage_epoch = [16,32,64,128]

    batch_size = 100

    num_classes = 55

    max_epoch = 200

    target_point_num = 2048

    ckpt = 'ckpt'

    sub_dir = 'submit'

    lr = 0.001

    current_w = 'current_w.pkl'

    best_w = 'best_w.pkl'

    lr_decay = 5


    temp_dir=os.path.join(root2,'temp')


config = Config()
