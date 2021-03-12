### Some Utils Functions

import matplotlib.pyplot as plt
import nltk
import re
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score, accuracy_score
from pycm import *
from sklearn.preprocessing import LabelEncoder

#### Preprocessing Function


# In[ ]:


# encode y_train
def encode_y(y_in):
    dic_y_mapping = {n:label for n,label in enumerate(np.unique(y_in))}
    inverse_dic = {v:k for k,v in dic_y_mapping.items()}
    y_in = np.array([inverse_dic[y] for y in y_in])
    return y_in, dic_y_mapping

def encode2_y(y_in):
    vectorizer = LabelEncoder()
    y_in = vectorizer.fit_transform(y_in)
    return y_in

'''
Preprocess a string.
:parameter
    :param text: string - name of column containing text
    :param lst_stopwords: list - list of stopwords to remove
    :param flg_stemm: bool - whether stemming is to be applied
    :param flg_lemm: bool - whether lemmitisation is to be applied
:return
    cleaned text
'''
def utils_preprocess_text(text, flg_stemm, flg_lemm, lst_stopwords):
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text


# In[ ]:


#### Metrics Functions


# In[ ]:


from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    return history.history['val_accuracy'][1]

def custom_metrics(matrix_arrays, class_name):
    
    # convert numpy matrix in dictionary
    new_matrix=np.zeros((7,7))
    matrix={}
    for i in range(7):
        linha={}
        for j in range(7):
            matrix_arrays[i,j]=int(np.round(matrix_arrays[i,j]))
            linha[j]=int(np.round(matrix_arrays[i,j]))
        matrix[i]=linha
    
    #plot confusion matrix
    plotCM(matrix_arrays, class_name, True, "Average - Artifact Classification")
    
    results = ConfusionMatrix(matrix=matrix)
    numClasses = len(class_name)
                
    tp = results.TP
    tn = results.TN
    fp = results.FP
    fn = results.FN
    
    macroMCC = 0.0
    macroAUC = 0.0
    microAUC = 0.0
    sum_tpfn = 0.0 
    for i in range(numClasses):

        tp_fn = tp[i] + fn[i]
        sum_tpfn = sum_tpfn + tp_fn
        
        sum1 = (tp[i] * tn[i] - fp[i] * fn[i])
        sum2 = ((tp[i] + fp[i])*(tp[i] + fn[i])*(tn[i] + fp[i])*(tn[i] + fn[i]))**0.5
        
        if type(results.MCC[i]) != float:
            results.MCC[i] = float(0)
        
        macroMCC = macroMCC + results.MCC[i]
        macroAUC = macroAUC + results.AUC[i]
        
        microAUC = microAUC + (tp_fn * results.AUC[i])

    macroMCC = macroMCC/numClasses
    macroAUC = macroAUC/numClasses
    microAUC = microAUC / sum_tpfn

    if sum2==0:
        microMCC = 0.0
    else:
        microMCC = sum1/sum2
        
    macro_ppv = results.overall_stat['PPV Macro']
    micro_ppv = results.overall_stat['PPV Micro']
    macro_recall= results.overall_stat['TPR Macro']
    micro_recall= results.overall_stat['TPR Micro']
    macro_f1 = results.overall_stat['F1 Macro']
    micro_f1 = results.overall_stat['F1 Micro']

    # plot metrics resume
    print(f'{"Metrics Resume":^98s}')
    print(f'{"="*98}')
    print(f'{"Class Name":<20}\t{"Precision":>10}\t{"Recall":>10}\t{"F1-Score":>10}\t{"MCC":>10}\t{"AUC":>10}')
    for i in range(numClasses):
        print(f'{class_name[i]:<20}\t{results.PPV[i]:>10.2}\t{results.TPR[i]:>10.2}\t{results.F1[i]:>10.2}\t{results.MCC[i]:>10.2}\t{results.AUC[i]:>10.2}')
    print()
    print(f'{"Macro Average":<20}\t{macro_ppv:>10.2}\t{macro_recall:>10.2}\t{macro_f1:>10.2}\t{macroMCC:>10.2}\t{macroAUC:>10.2}')
    print(f'{"Micro Average":<20}\t{micro_ppv:>10.2}\t{micro_recall:>10.2}\t{micro_f1:>10.2}\t{microMCC:>10.2}\t{microAUC:>10.2}')
    
def confusionMatrix(real, prediction):
    classes = np.unique(real)
    nClasses = len(classes)
    cm = np.zeros((nClasses, nClasses))
    cmNorm = []
    print(cmNorm)
    for i in range(len(real)):
        indexReal, = np.where(classes == real[i])
        indexReal = indexReal[0]
        indexPred, = np.where(classes == prediction[i])
        indexPred = indexPred[0]
        cm[indexReal][indexPred] = cm[indexReal][indexPred] + 1
    cmNorm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.array(cm,dtype=int)
    return cm, cmNorm

def plotCM(cm, classes,normalize=False,title=None,cmap=plt.cm.cool):
  
    if normalize == True:
      title = "Normalized"+title
    else:
      title = "No normalized"+title
      
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),

           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

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
    return ax

