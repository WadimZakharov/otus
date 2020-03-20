import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 60)

import itertools

import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn-colorblind')
plt.rc('font', size=14)

#import lightgbm as lgb
#import catboost as cb
from sklearn import model_selection
from sklearn.metrics import brier_score_loss,matthews_corrcoef,roc_curve, precision_recall_curve, auc, cohen_kappa_score, classification_report, mean_squared_error, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

def print_metrics(X_test,X_train,Y_test,Y_train,trashhold,clf=None,prnt=1):
    Y_train_pred = None
    if clf == None:
        Y_test_pred = X_test
        Y_train_pred = X_train
    else:
        Y_test_pred = clf.predict(X_test)
        if (len(X_train) > 1) and (len(Y_train) > 1):
            Y_train_pred = clf.predict(X_train)
    if prnt == 1:
        if (len(X_train) > 1) and (len(Y_train) > 1):
            print('Точность на обучающей выборке: ',accuracy_score(Y_train, (Y_train_pred > trashhold)))
            print('Precision на обучающей выборке: ',precision_score(Y_train, (Y_train_pred > trashhold)))
            print('Recall на обучающей выборке: ',recall_score(Y_train, (Y_train_pred > trashhold)))
            print('F-мера на обучающей выборке: ',f1_score(Y_train, (Y_train_pred > trashhold)))
        print('Точность на тестовой выборке: ',accuracy_score(Y_test, (Y_test_pred > trashhold)))
        print('Precision на тестовой выборке: ',precision_score(Y_test, (Y_test_pred > trashhold)))
        print('Recall на тестовой выборке: ',recall_score(Y_test, (Y_test_pred > trashhold)))
        print('F-мера на тестовой выборке: ',f1_score(Y_test, (Y_test_pred > trashhold)))
    return Y_test_pred,Y_train_pred


def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(1/n_samples, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
def print_report(y_test, y_pred, is_multiclass:bool = False, thresh:float = 0.1, classes:list = ['Non-paid', 'Paid']):
    
    if is_multiclass:
        ind = np.array([np.argmax(x) for x in y_pred])
        print('Accuracy is:', accuracy_score(y_test, ind))
    else:
        ind = np.array([1 if x >=thresh else 0 for x in y_pred])

    print(f"Sample percent for sending to event: {len(ind[ind != 0])/len(ind)}")
    print(f"Cohen's kappa score is: {cohen_kappa_score(y_test, ind)}")
    report = classification_report(y_test, ind, target_names=classes)
    print(report)

    cnf_matrix = confusion_matrix(y_test, ind)

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes, normalize=False, title='Not normalized confusion matrix')

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True, title='Normalized confusion matrix')
    
def plot_roc_curve(y_test, y_pred, ax=None) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    if ax is None:
        ax = plt.gca()
        
    ax.set_title('Receiver Operating Characteristic')
    ax.plot(false_positive_rate, true_positive_rate, 'b',label=f'AUC = {roc_auc:.2f}')
    ax.legend(loc='lower right')
    ax.plot([0,1],[0,1],'r--')
    ax.set_xlim([-0.1,1.2])
    ax.set_ylim([-0.1,1.2])
    ax.grid()
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    return(roc_auc)

def plot_pr_curve(y_test, y_pred, ax=None) -> float:
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)
    
    if ax is None:
        ax = plt.gca()
    
    ax.set_title('Precision-Recall Curve')
    ax.plot(recall, precision, 'b',label=f'PR AUC = {pr_auc:.2f}')
    ax.legend(loc='upper right')
    ax.plot([0,1],[0.5,0.5],'r--')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.grid()
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    return(pr_auc)

def plot_f1_curve(y_test, y_pred, ax=None) -> (float, float,float,float):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    F1 = 2 * (precision * recall) / (precision + recall)
    maxf1 = np.max(F1)
    maxf1thr = thresholds[np.argmax(F1)]
    best_f1_pr = precision[np.argmax(F1)]
    best_f1_re = recall[np.argmax(F1)]
        
    if ax is None:
        ax = plt.gca()
    
    ax.set_title('F1 Curve')
    ax.plot(thresholds, F1, 'b',label=f'max F1 = {maxf1:.2f}')
    ax.legend(loc='upper right')
    ax.plot([0,1],[0.5,0.5],'r--')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.grid()
    ax.set_ylabel('f1')
    ax.set_xlabel('thresholds')
    return(float("{0:.3f}".format(maxf1)),float("{0:.2f}".format(maxf1thr)),
           float("{0:.3f}".format(best_f1_pr)),float("{0:.2f}".format(best_f1_re)))

def plot_cohen_kappa(y_test, y_pred, ax=None) -> (float, float):
    thresholds = np.linspace(0,1,100)
    kappa=[]
    for thr in thresholds:
        ind = np.array([1 if x >=thr else 0 for x in y_pred])
        kappa.append(cohen_kappa_score(y_test, ind))

    kappa = np.array(kappa)
    max_k = np.max(kappa)
    max_thr = thresholds[np.argmax(kappa)]
    
    if ax is None:
        ax = plt.gca()
    
    ax.set_title("Cohen's kappa score curve")
    ax.plot(thresholds, kappa, 'b', label=f'max kappa is {max_k:.2f}')
    ax.legend(loc='upper right')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.grid()
    ax.set_ylabel('kappa')
    ax.set_xlabel('threshold')
    return(float("{0:.3f}".format(max_k)),float("{0:.2f}".format(max_thr)))

def plot_matthews_corrcoef(y_test, y_pred, ax=None) -> (float, float):
    thresholds = np.linspace(0,1,100)
    mcc=[]
    for thr in thresholds:
        ind = np.array([1 if x >=thr else 0 for x in y_pred])
        mcc.append(matthews_corrcoef(y_test, ind))

    mcc = np.array(mcc)
    max_mcc = np.max(mcc)
    max_thr = thresholds[np.argmax(mcc)]
    
    if ax is None:
        ax = plt.gca()
    
    ax.set_title("Matthews correlation coefficient curve")
    ax.plot(thresholds, mcc, 'b', label=f'max MCC is {max_mcc:.2f}')
    ax.legend(loc='upper right')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.grid()
    ax.set_ylabel('kappa')
    ax.set_xlabel('threshold')
    return(float("{0:.3f}".format(max_mcc)),float("{0:.2f}".format(max_thr)))

def plot_brier_scor(y_test, y_pred, ax=None) -> (float, float):
    thresholds = np.linspace(0,1,100)
    brier=[]
    for thr in thresholds:
        ind = np.array([1 if x >=thr else 0 for x in y_pred])
        brier.append(brier_score_loss(y_test, ind))

    brier = np.array(brier)
    min_brier = np.min(brier)
    br_thr = thresholds[np.argmin(brier)]
    
    if ax is None:
        ax = plt.gca()
    
    ax.set_title("Brier score curve")
    ax.plot(thresholds, brier, 'b', label=f'min brier is {min_brier:.2f}')
    ax.legend(loc='upper right')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.grid()
    ax.set_ylabel('brier')
    ax.set_xlabel('threshold')
    return(float("{0:.3f}".format(min_brier)),float("{0:.2f}".format(br_thr)))

def plot_rel_probs(y_test, y_pred, n:int = 1000, ax=None):
    t_df = pd.DataFrame(data={'scor': y_pred, 'real': y_test})
    t_df.sort_values(by=['scor'],inplace=True)
    
    parts = np.array_split(t_df.values, n)
    parts = np.mean(parts, axis=1, keepdims=True)
    parts = np.reshape(parts, (n, 2))

    if ax is None:
        ax = plt.gca()
          
    ax.set_title("Concordance of model predictions with prior probabilities")
    ax.plot(parts[:,0], parts[:,1], 'bo', label=f"pred. lims are [{t_df.scor.iloc[0]:.5f}, {t_df.scor.iloc[-1]:.5f}]")
    #ax.legend(loc='lower right')
    x = np.linspace(0, 1, 3)
    ax.plot(x, x, 'r')
    ax.set_xlabel('Ответ алгоритма')
    ax.set_ylabel('Оценка')
    
def plot_metrics(y_test, y_pred,n):
    """
    This function plots all metrics.
    """
    
    nrows, ncols = 2, 3
    it1, it2 = itertools.tee(range(nrows*ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(21, 12))
        
    roc_auc = plot_roc_curve(y_test, y_pred, ax=axes[next(it1) // ncols, next(it2) % ncols])
    pr_auc = plot_pr_curve(y_test, y_pred, ax=axes[next(it1) // ncols, next(it2) % ncols])
    plot_rel_probs(y_test, y_pred,n, ax=axes[next(it1) // ncols, next(it2) % ncols])
    
    max_mcc, mcc_thr = plot_matthews_corrcoef(y_test, y_pred, ax=axes[next(it1) // ncols, next(it2) % ncols])
    max_k, kappa_thr = plot_cohen_kappa(y_test, y_pred, ax=axes[next(it1) // ncols, next(it2) % ncols])
    min_brier, br_thr = plot_brier_scor(y_test, y_pred, ax=axes[next(it1) // ncols, next(it2) % ncols])
    #fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(21, 12))
       
    #maxf1, maxf1thr,bpr,bre = plot_f1_curve(y_test, y_pred, ax=axes)
    plt.show()
    print(f"The rmse of model's prediction is: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"The Gini of model's prediction is: {Gini(y_test, y_pred):.4f}")
    print(f"The ROC AUC of model's prediction is: {roc_auc:.4f}")
    print(f"The PR AUC of model's prediction is: {pr_auc:.4f}")
    print(f"Max Cohen's kappa is {max_k:.3f} with threshold = {kappa_thr:.2f}")
    print(f"Max Matthews correlation coefficient is {max_mcc:.3f} with threshold = {mcc_thr:.2f}")
    print(f"Min Brier score is {min_brier:.3f} with threshold = {br_thr:.2f}")
    #print(f"Max F1 score is {maxf1:.3f} with threshold = {maxf1thr:.2f}. Precision = {bpr:.2f}, recall = {bre:.2f}")