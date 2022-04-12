from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, roc_auc_score
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve

def yuedeng_index(tpr,fpr):
    a=0
    ii,jj=0,0
    for i,j in zip(tpr,fpr):
       if i-j>=a:
           a=i-j
           ii=i
           jj=j
    print('Cutoff_sens:',ii,'Cutoff_spec:',1-jj,'Cutoff-point:(%.2f,%.2f)'%(ii,jj))
    return ii,jj

def draw_PR(recall, precision,model_name,k):
    plt.figure(k)
    lw = 2
    print(recall,precision)
    plt.plot(recall, precision,color='red',lw=lw)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('PR curve for ' + model_name)
    plt.savefig(r'/home/jiangsiqing/ml/roccurve/vn/v%i/PR/%s_PR curve.jpg' % (int(k + 1), model_name))
    plt.close()

def draw_roc_curve(train_pre_proba, test_pre_proba, train_auc, test_auc, model_name, k):
    fpr, tpr, roc_auc = train_pre_proba
    test_fpr, test_tpr, test_roc_auc = test_pre_proba
    y, x = yuedeng_index(test_tpr, test_fpr)
    plt.figure(1)
    lw = 2


    plt.scatter([x], [y], s=45,c='darkviolet',marker='*')
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve for train(area = %0.2f)' % train_auc)
    plt.plot(test_fpr, test_tpr, color='red',
             lw=lw, label='ROC curve for test(area = %0.2f)' % test_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.hlines(y, -0.1, x, colors='r', linestyle="--")
    # plt.text(-0.04, y, str(y))
    plt.vlines(x,-0.1,y,colors='r', linestyle='--')
    plt.text(x, y+0.01, '(%s,%s)'%(str(round(x,3)),str(round(y,3))))
    plt.annotate("Max Youden index(Cutoff-Value)", (x, y), xycoords='data',
                 xytext=(0.5,0.3),
                 arrowprops=dict(arrowstyle='->'))
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Positive Rate(TPR)')
    plt.title('Roc curve(%s)'%model_name)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(r'/home/jiangsiqing/ml/roccurve/vn/v%i/%s_roc.jpg' % (int(k + 1), model_name))
    plt.close()


a = pd.read_csv(r'/home/jiangsiqing/ml/roccurve/360078.csv').values
x = a[:, 45:70]
# print(x)
for k in range(74, 78):
    print('k=%i' % k)
    y = np.array(a[:, k])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=70)
    print('split ok!!!!!!')
    LR = LogisticRegression(max_iter=3000)
    LR.fit(x_train, y_train)
    print('lr fit ok!!')
    y_train_pred = LR.predict_proba(x_train)[:, 1]
    y_test_pred = LR.predict_proba(x_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, y_train_pred)
    train_roc = roc_curve(y_train, y_train_pred)
    test_roc = roc_curve(y_test, y_test_pred)

    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)

    draw_roc_curve(train_roc, test_roc, train_auc, test_auc, 'LogisticsRegression', k)
    draw_PR(recall,precision,'LogisticsRegression',k)
    print('************lr over***********')

    DT = DecisionTreeClassifier(max_depth=6)
    DT.fit(x_train, y_train)
    print('DT fit ok!!')
    y_train_pred = DT.predict_proba(x_train)[:, 1]
    y_test_pred = DT.predict_proba(x_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, y_train_pred)
    train_roc = roc_curve(y_train, y_train_pred)
    test_roc = roc_curve(y_test, y_test_pred)

    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)

    draw_roc_curve(train_roc, test_roc,train_auc, test_auc, 'Decision Tree', k)
    draw_PR(recall,precision,'Decision Tree',k)
    print('************dt over***********')

    RF = RandomForestClassifier()
    RF.fit(x_train, y_train)
    print('RF fit ok!!')
    y_train_pred = RF.predict_proba(x_train)[:, 1]
    y_test_pred = RF.predict_proba(x_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, y_train_pred)
    train_roc = roc_curve(y_train, y_train_pred)
    test_roc = roc_curve(y_test, y_test_pred)

    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)

    draw_roc_curve(train_roc, test_roc, train_auc, test_auc, 'Random Forests', k)
    draw_PR(recall,precision,'Random Forests',k)
    print('************rf over***********')

    XGB = XGBClassifier(eval_metric=['logloss', 'auc', 'error'], use_label_encoder=False)
    XGB.fit(x_train, y_train)
    print('XGB fit ok!!')
    y_train_pred = XGB.predict_proba(x_train)[:, 1]
    y_test_pred = XGB.predict_proba(x_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, y_train_pred)
    train_roc = roc_curve(y_train, y_train_pred)
    test_roc = roc_curve(y_test, y_test_pred)

    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)

    draw_roc_curve(train_roc, test_roc, train_auc, test_auc, 'XGBoost', k)
    draw_PR(recall,precision,'XGBoost',k)
    print('************xgb_over***********')

    svc = SVC(kernel='linear',probability=True)
    svc.fit(x_train,y_train)
    print('svc fit ok!!')
    y_train_pred = svc.predict_proba(x_train)[:, 1]
    y_test_pred = svc.predict_proba(x_test)[:, 1]
    precision,recall,thresholds=precision_recall_curve(y_train,y_train_pred)
    train_roc = roc_curve(y_train, y_train_pred)
    test_roc = roc_curve(y_test, y_test_pred)

    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)

    draw_roc_curve(train_roc, test_roc,train_auc, test_auc, 'SVC',k)
    draw_PR(recall,precision,'SVC',k)
    print('************svc over***********')