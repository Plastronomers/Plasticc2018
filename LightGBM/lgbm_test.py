import os
os.environ['KMP_WARNINGS'] = 'off'
from sklearn.externals import joblib

import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.utils import to_categorical

clf_load = joblib.load('./saved_models/saved_model_lgbm2.pkl')

test = pd.read_csv('./Pre_Full_Features_test.csv')
print('reading done!')
#test = test['target'].replace({991:99 , 992:99 , 993:99 , 994:99})
test = test[test.target != 991]
test = test[test.target != 992]
test = test[test.target != 993]
test = test[test.target != 994]

y_test = test['target']
x_test = test.drop(['Unnamed: 0','target'],axis = 1)

clms = []
with open('./information/clm.txt', 'r') as fh:
    for clm in fh:
        clm = clm.rstrip("\n")
        clms.append(clm)

x_test = x_test.drop(clms,axis = 1)

y_pred = clf_load.predict(x_test)
y_pred1 = clf_load.predict_proba(x_test)
print(y_pred1)

classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}

y_p_norm = np.clip(a=y_pred1, a_min=1e-15, a_max=1-1e-15)

unique_y = np.unique(y_test)
print(unique_y)
print(len(y_test))

class_map = dict()
for i,val in enumerate(unique_y):
    class_map[val] = i       
y_map = np.zeros((y_test.shape[0],))
y_map = np.array([class_map[val] for val in y_test])
y_categorical = to_categorical(y_map)
y_oh = y_categorical

N_arr = np.zeros(shape=len(classes))
for i,val in enumerate(classes):
    s = np.sum(y_test==val)
    N_arr[i]=s
N_arr

f = 0
w = 0
N_obj = len(x_test)
for i,val in enumerate(classes):
    wi = class_weight[val]
    w += wi
    for j in range(len(y_test)):
        f += (wi*y_oh[j,i]*np.log(y_p_norm[j,i]))/N_arr[i]

Log_loss = -f/w

print(Log_loss)


w = y_test.value_counts()
weights = {i : np.sum(w) / w[i] for i in w.index}

def multi_weighted_logloss(y_true, y_preds):

    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

print(multi_weighted_logloss(y_test, y_pred1))
#oof_preds = np.zeros((len(test), np.unique(y_test).shape[0]))
#pr = pd.DataFrame(y_pred)
#pr.to_csv('./preds_lgbm.csv')

print(accuracy_score(y_test, y_pred, normalize = True))

#cm
def conf_plotter(y_true, y_pred, classes,
                      normalize=0,
                      title=None,
                      cmap=plt.cm.pink_r,
                      mode = 'train',
                      save = 0,
                      name = 'conf.jpg'):

    if not title:
        if mode == 'train':
            title = 'Normalized confusion matrix on train dataset'
        else:
            title = 'Normalized confusion matrix on test dataset'
            

    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax ,fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           xlabel='Predicted label',
           ylabel='True label')
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] ):
                    
        item.set_fontsize(15)
     
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        
        item.set_fontsize(10)

        
                
    plt.gca().invert_yaxis()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if save == 1:
        plt.savefig(name)
    return ax

classes = ['class_6','class_15','class_16','class_42','class_52','class_53','class_62','class_64','class_65','class_67','class_88','class_90','class_92','class_95']

labels = [classes[i] for i in range(len(classes))]
conf_plotter(y_test , y_pred , labels , normalize=1 , mode ='test',cmap = plt.cm.Oranges,save=0 , name='test_lgbm.jpg')
plt.savefig('Normalized confusion matrix on test dataset_lgbm model')

print('yes!!!')
