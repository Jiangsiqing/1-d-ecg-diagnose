from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,roc_auc_score
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

def Find_Optimal_Cutoff(TPR,FPR,threshold):
    y=TPR-FPR
    yuedeng_index=np.argmax(y)
    opt_threshold=threshold[yuedeng_index]
    point=[FPR[yuedeng_index],TPR[yuedeng_index]]
    return opt_threshold,point
    
def draw_roc_curve(train_pre_proba, test_pre_proba,threshold1,threshold2, train_auc, test_auc, model_name,k):
    fpr, tpr, roc_auc = train_pre_proba
    test_fpr, test_tpr, test_roc_auc = test_pre_proba
    opt_threshold1, point1=Find_Optimal_Cutoff(TPR=tpr,FPR=fpr,threshold=threshold1)
    opt_threshold2, point2=Find_Optimal_Cutoff(TPR=tpr,FPR=fpr,threshold=threshold2)
    plt.figure()
    plt.plot(point1[0],point1[1],marker='o',color='y')
    plt.text(point1[0],point1[1],f'Threshold:{opt_threshold1:.2f}')
    plt.plot(point2[0],point2[1],marker='o',color='y')
    plt.text(point2[0],point2[1],f'Threshold:{opt_threshold2:.2f}')


    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve for train(area = %0.2f)' % train_auc)
    plt.plot(test_fpr, test_tpr, color='red',
             lw=lw, label='ROC curve for test(area = %0.2f)' % test_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Positive Rate(TPR)')
    plt.title('Roc curve for '+model_name)
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(r'/home/jiangsiqing/ml/roccurve/v%i/%s_roc.jpg'%(int(k+1),model_name))
a=pd.read_csv(r'/home/jiangsiqing/ml/roccurve/400078.csv').values
x=a[:,:70]
for k in range(74,78):
    print('k=%i'%k)
    y=np.array(a[:,k])
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=70)
    print('split ok!!!!!!')


    LR = LogisticRegression(max_iter=3000)
    LR.fit(x_train,y_train)
    print('lr fit ok!!')
    _,_,threshold1=metrics.roc_curve(y_train,LR.predict(x_train))
    _,_,threshold2=metrics.roc_curve(y_test,LR.predict(x_test))
    y_train_pred = LR.predict_proba(x_train)[:, 1]
    y_test_pred = LR.predict_proba(x_test)[:, 1]

    train_roc = roc_curve(y_train, y_train_pred)
    test_roc = roc_curve(y_test, y_test_pred)

    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)

    draw_roc_curve(train_roc, test_roc,threshold1,threshold2, train_auc, test_auc, 'LR',k)
    print('************lr over***********')



    DT = DecisionTreeClassifier(max_depth=6)
    DT.fit(x_train,y_train)
    print('DT fit ok!!')
    _,_,threshold1=metrics.roc_curve(y_train,DT.predict(x_train))
    _,_,threshold2=metrics.roc_curve(y_test,DT.predict(x_test))    
    y_train_pred = DT.predict_proba(x_train)[:, 1]
    y_test_pred = DT.predict_proba(x_test)[:, 1]

    train_roc = roc_curve(y_train, y_train_pred)
    test_roc = roc_curve(y_test, y_test_pred)

    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)

    draw_roc_curve(train_roc, test_roc, threshold1,threshold2,train_auc, test_auc, 'DT',k)
    print('************dt over***********')


    RF = RandomForestClassifier()
    RF.fit(x_train,y_train)
    print('RF fit ok!!')
    _,_,threshold1=metrics.roc_curve(y_train,RF.predict(x_train))
    _,_,threshold2=metrics.roc_curve(y_test,RF.predict(x_test))    
    print(RF.predict(x_train),RF.predict_proba(x_train)[:, 1])
    y_train_pred = RF.predict_proba(x_train)[:, 1]
    y_test_pred = RF.predict_proba(x_test)[:, 1]

    train_roc = roc_curve(y_train, y_train_pred)
    test_roc = roc_curve(y_test, y_test_pred)

    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)

    draw_roc_curve(train_roc, test_roc, threshold1,threshold2,train_auc, test_auc, 'RF',k)
    print('************rf over***********')


    XGB = XGBClassifier(eval_metric=['logloss','auc','error'],use_label_encoder=False)
    XGB.fit(x_train,y_train)
    print('XGB fit ok!!')
    _,_,threshold1=metrics.roc_curve(y_train,XGB.predict(x_train))
    _,_,threshold2=metrics.roc_curve(y_test,XGB.predict(x_test))    
    y_train_pred = XGB.predict_proba(x_train)[:, 1]
    y_test_pred = XGB.predict_proba(x_test)[:, 1]

    train_roc = roc_curve(y_train, y_train_pred)
    test_roc = roc_curve(y_test, y_test_pred)

    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)

    draw_roc_curve(train_roc, test_roc, threshold1,threshold2,train_auc, test_auc, 'XGB',k)
    print('************xgb_over***********')

    #svc = SVC(kernel='linear',probability=True)
    #svc.fit(x_train,y_train)
    #print('svc fit ok!!')
    #_,_,threshold1=metrics.roc_curve(y_train,svc.predict(x_train))
    #_,_,threshold2=metrics.roc_curve(y_test,svc.predict(x_test))    
    #y_train_pred = svc.predict_proba(x_train)[:, 1]
    #y_test_pred = svc.predict_proba(x_test)[:, 1]

    #train_roc = roc_curve(y_train, y_train_pred)
    #test_roc = roc_curve(y_test, y_test_pred)

    #train_auc = roc_auc_score(y_train, y_train_pred)
    #test_auc = roc_auc_score(y_test, y_test_pred)

    #draw_roc_curve(train_roc, test_roc,threshold1,threshold2, train_auc, test_auc, 'SVC',k)
    #print('************svc over***********')
# models = [LR,svc,DT,RF,XGB]
#
# names = ["LR","SVC", 'DT', "RF","Xgb"]
# evaluates = ['accuracy','precision','recall','f1','auc']
# for name, model in zip(names, models):


