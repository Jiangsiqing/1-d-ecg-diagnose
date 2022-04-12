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
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix





def draw_PR(recall, precision,model_name,k):
    plt.figure(k)
    lw = 2
    # print(recall,precision)
    plt.plot(recall, precision,color='red',lw=lw)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('PR curve for ' + model_name)
    plt.savefig(r'/home/jiangsiqing/ml/roccurve/v%i/PR/%s_PR curve.jpg' % (int(k + 1), model_name))
    plt.close()

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
    
def draw_roc_curve(train_pre_proba, test_pre_proba, train_auc, test_auc, model_name,k):
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
    plt.title('Roc curve for '+model_name)
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(r'/home/jiangsiqing/ml/roccurve/v%i/%s_roc.jpg'%(int(k+1),model_name))
    plt.close()
a=pd.read_csv(r'/home/jiangsiqing/ml/roccurve/360078.csv').values


x=a[:,:44]
print(x.shape)



print('*************ROC*************')
k=76

y=np.array(a[:,k])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print('split ok!!!!!!')



#LogisticRegression
# LR = LogisticRegression(max_iter=3000)

# param_dist_lr={"penalty": 'l2', "dual": False, "tol": 0.0001, "fit_intercept": True,
#  "class_weight": 'auto', "solver": 'newton-cg',
#  "max_iter": 1000, "multi_class": "ovr", "verbose": 0, "warm_start": False, "n_jobs": 8,
#  "l1_ratio": None}
# lr_grid=GridSearchCV(LR,param_dist_lr,cv=3,scoring='neg_log_loss',n_jobs=-1)
# lr_grid.fit(x_train,y_train)
# LR.fit(x_train,y_train)
# print('lr fit ok!!')
# y_train_pred = LR.predict_proba(x_train)[:, 1]
# y_test_pred = LR.predict_proba(x_test)[:, 1]
# precision,recall,thresholds=precision_recall_curve(y_train,y_train_pred)
# train_roc = roc_curve(y_train, y_train_pred)
# test_roc = roc_curve(y_test, y_test_pred)
#
# train_auc = roc_auc_score(y_train, y_train_pred)
# test_auc = roc_auc_score(y_test, y_test_pred)
#
# draw_roc_curve(train_roc, test_roc,train_auc, test_auc, 'LogisticsRegression',k)
# draw_PR(recall, precision, 'LogisticsRegression', k)
# prediction1=LR.predict(x_test)
# print('acc', accuracy_score(y_test, prediction1))
# print('f1', f1_score(y_test, prediction1))
# print('cm', confusion_matrix(y_test, prediction1))
# print('recall', recall_score(y_test, prediction1))
# print('pre', precision_score(y_test, prediction1))
# print('************lr over***********')
#
#
#
#
# #Decision Tree
#
# DT = DecisionTreeClassifier(max_depth=6)
# DT.fit(x_train,y_train)
# print('Decision Tree fit ok!!')
# y_train_pred = DT.predict_proba(x_train)[:, 1]
# y_test_pred = DT.predict_proba(x_test)[:, 1]
# precision,recall,thresholds=precision_recall_curve(y_train,y_train_pred)
# train_roc = roc_curve(y_train, y_train_pred)
# test_roc = roc_curve(y_test, y_test_pred)
#
# train_auc = roc_auc_score(y_train, y_train_pred)
# test_auc = roc_auc_score(y_test, y_test_pred)
#
# draw_roc_curve(train_roc, test_roc,train_auc, test_auc, 'Decision Tree',k)
# draw_PR(recall, precision, 'Decision Tree', k)
# prediction2=DT.predict(x_test)
# print('acc', accuracy_score(y_test, prediction2))
# print('f1', f1_score(y_test, prediction2))
# print('cm', confusion_matrix(y_test, prediction2))
# print('recall', recall_score(y_test, prediction2))
# print('pre', precision_score(y_test, prediction2))
# print('************Decision Tree over***********')
#
#
#
#
#
# #Random Forests
# param_dist_rf= {"min_samples_split": 10, "min_samples_leaf": 20,
#                 "max_depth": 8, "max_features": 'sqrt',
#    }
#
# RF = RandomForestClassifier()
# RF.fit(x_train,y_train)
# print('RF fit ok!!')
# y_train_pred = RF.predict_proba(x_train)[:, 1]
# y_test_pred = RF.predict_proba(x_test)[:, 1]
# precision,recall,thresholds=precision_recall_curve(y_train,y_train_pred)
# train_roc = roc_curve(y_train, y_train_pred)
# test_roc = roc_curve(y_test, y_test_pred)
#
# train_auc = roc_auc_score(y_train, y_train_pred)
# test_auc = roc_auc_score(y_test, y_test_pred)
#
# draw_roc_curve(train_roc, test_roc,train_auc, test_auc, 'Random Forests',k)
# draw_PR(recall, precision, 'Random Forests', k)
# prediction3=RF.predict(x_test)
# print('acc', accuracy_score(y_test, prediction3))
# print('f1', f1_score(y_test, prediction3))
# print('cm', confusion_matrix(y_test, prediction3))
# print('recall', recall_score(y_test, prediction3))
# print('pre', precision_score(y_test, prediction3))
# print('************rf over***********')
#
#
#
#
# # XGBClassifier
#
param_dist_xgb = {
    'n_estimators': [577],
    'max_depth': [5],
    'learning_rate': [0.001],
    'subsample': [0.8],


}

# 'max_depth': [6],
# 'learning_rate': [0.3],
# 'subsample': [0.8],
# param_dist_xgb = {
#     'n_estimators': range(80, 100, 2),
#     'max_depth': range(2, 4, 1),
#     'learning_rate': np.linspace(0.01, 2, 2),
#     'subsample': np.linspace(0.7, 0.9, 2),
#     'colsample_bytree': np.linspace(0.5, 0.98, 1),
#     'min_child_weight': range(1, 3, 1)
# }
#
XGB = XGBClassifier(eval_metric=['logloss','auc','error'],use_label_encoder=False)
xgb_grid=GridSearchCV(XGB,param_dist_xgb,cv=2,scoring='neg_log_loss')
xgb_grid.fit(x_train,y_train)
print('best_estimator:',xgb_grid.best_estimator_)
print('feature_important:',xgb_grid.feature_importances_)
# XGB.fit(x_train,y_train)
print('XGBoost fit ok!!')
y_train_pred = xgb_grid.predict_proba(x_train)[:, 1]
y_test_pred = xgb_grid.predict_proba(x_test)[:, 1]
precision,recall,thresholds=precision_recall_curve(y_train,y_train_pred)
train_roc = roc_curve(y_train, y_train_pred)
test_roc = roc_curve(y_test, y_test_pred)

train_auc = roc_auc_score(y_train, y_train_pred)
test_auc = roc_auc_score(y_test, y_test_pred)

draw_roc_curve(train_roc, test_roc,train_auc, test_auc, 'XGBoost',k)
draw_PR(recall, precision, 'XGBoost', k)

prediction4=xgb_grid.predict(x_test)
print(prediction4,y_test)
print('acc', accuracy_score(y_test, prediction4))
print('f1', f1_score(y_test, prediction4))
print('cm', confusion_matrix(y_test, prediction4))
print('recall', recall_score(y_test, prediction4))
print('pre', precision_score(y_test, prediction4))
print('************XGBoost_over***********')
#



#SVC

# svc = SVC(kernel='linear',probability=True)
# svc.fit(x_train,y_train)
# print('svc fit ok!!')
# y_train_pred = svc.predict_proba(x_train)[:, 1]
# y_test_pred = svc.predict_proba(x_test)[:, 1]
# precision,recall,thresholds=precision_recall_curve(y_train,y_train_pred)
# train_roc = roc_curve(y_train, y_train_pred)
# test_roc = roc_curve(y_test, y_test_pred)
#
# train_auc = roc_auc_score(y_train, y_train_pred)
# test_auc = roc_auc_score(y_test, y_test_pred)
#
# draw_roc_curve(train_roc, test_roc, train_auc, test_auc, 'SVC',k)
# draw_PR(recall, precision, 'SVC', k)
# prediction5=svc.predict(x_test)
# print('acc', accuracy_score(y_test, prediction5))
# print('f1', f1_score(y_test, prediction5, average='macro'))
# print('cm', confusion_matrix(y_test, prediction5))
# print('recall', recall_score(y_test, prediction5, average='macro'))
# print('pre', precision_score(y_test, prediction5, average='macro'))
# print('************svc over***********')



#######################################################
# print('************OLGA***********')
# y=np.array(a[:,73])
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=25)
# y_train-=1
# y_test-=1
#
# # LR = LogisticRegression(max_iter=3000)
# DT = DecisionTreeClassifier(max_depth=2)
# RF = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
# XGB = XGBClassifier(eval_metric=['logloss','auc','error'],use_label_encoder=False)
#
# # svc = SVC(kernel='linear',probability=True)
# LR.fit(x_train,y_train)
# DT.fit(x_train,y_train)
# RF.fit(x_train,y_train)
# XGB.fit(x_train,y_train)
# xgb_grid=GridSearchCV(XGB,param_dist_xgb,cv=5,scoring='neg_log_loss')
# xgb_grid.fit(x_train,y_train)
# print(xgb_grid.best_estimator_)
# # svc.fit(x_train,y_train)
# pre1,pre2,pre3,pre4=LR.predict(x_test),DT.predict(x_test),RF.predict(x_test),xgb_grid.predict(x_test)
#
#
#
# print('**********1.LR-4**********')
# print('acc', accuracy_score(y_test,pre1))
# print('f1', f1_score(y_test, pre1, average='macro'))
# print('cm', confusion_matrix(y_test, pre1))
# print('recall', recall_score(y_test, pre1, average='macro'))
# print('pre', precision_score(y_test, pre1, average='macro'))
# print('**********1.LR-4 OVER**********')
#
# print('**********2.DT-4**********')
# print('acc', accuracy_score(y_test,pre2))
# print('f1', f1_score(y_test, pre2, average='macro'))
# print('cm', confusion_matrix(y_test, pre2))
# print('recall', recall_score(y_test, pre2, average='macro'))
# print('pre', precision_score(y_test, pre2, average='macro'))
# print('**********2.DT-4 OVER**********')
#
# print('**********3.RF-4**********')
# print('acc', accuracy_score(y_test,pre3))
# print('f1', f1_score(y_test, pre3, average='macro'))
# print('cm', confusion_matrix(y_test, pre3))
# print('recall', recall_score(y_test, pre3, average='macro'))
# print('pre', precision_score(y_test, pre3, average='macro'))
# print('**********3.RF-4 OVER**********')
#
# print('**********4.XGB-4**********')
# print('acc', accuracy_score(y_test,pre4))
# print('f1', f1_score(y_test, pre4, average='macro'))
# print('cm', confusion_matrix(y_test, pre4))
# print('recall', recall_score(y_test, pre4, average='macro'))
# print('pre', precision_score(y_test, pre4, average='macro'))
# print('**********4.XGB-4 OVER**********')

# print('**********5.svc-4**********')
# print('acc', accuracy_score(y_test,pre5))
# print('f1', f1_score(y_test, pre5, average='macro'))
# print('cm', confusion_matrix(y_test, pre5))
# print('recall', recall_score(y_test, pre5, average='macro'))
# print('pre', precision_score(y_test, pre5, average='macro'))
# print('**********5.svc-4 OVER**********')


