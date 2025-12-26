import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import math
from os.path import isfile, join
from os import listdir
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, jaccard_score

#Good guide
#https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
def to_percent(value):
    return value*100

def space():
    print('\n\n')



def metrics_f1(y_true,y_pred,f1_scores):
    f1_scores[0] += f1_score(y_true, y_pred,average='macro')
    f1_scores[1] += f1_score(y_true, y_pred,average='weighted')
    '''
    print('jaccard index macro:')
    print(jaccard_score(y_true, y_pred,average='macro'))
    print('jaccard index weighted:')
    print(jaccard_score(y_true, y_pred,average='weighted'))
    '''
    return f1_scores

def metrics_jaccard(y_true,y_pred,jaccard_scores):
    jaccard_scores[0] += jaccard_score(y_true, y_pred,average='macro')
    jaccard_scores[1] += jaccard_score(y_true, y_pred,average='weighted')
    '''
    print('jaccard index macro:')
    print(jaccard_score(y_true, y_pred,average='macro'))
    print('jaccard index weighted:')
    print(jaccard_score(y_true, y_pred,average='weighted'))
    '''
    return jaccard_scores

def read_files(files,pred_dir,gt_folder):
    #for i in files['name']:
    jaccard_scores = [0,0]
    f1_scores = [0,0]
    num = 0
    for i in files:
        if(i%100 == 0):
            print(i)
        
        y_true = cv2.imread(gt_folder + 'ground_truth_training_' + str(i) + '.png')
        y_true = y_true[:,:,0]
        #y_pred = plt.imread('Predictions/Test_2_Rahel/training_' + str(i) + '.TIF')
        #y_pred = plt.imread('../../Final_cnn/5_fold/predictions/training_' + str(i) + '.TIF')
        y_pred = cv2.imread(pred_dir+ str(i) + '.png')
        y_pred = y_pred[:,:,0]
        y_pred = np.where(y_pred==85,60,y_pred)
        y_pred = np.where(y_pred==170,255,y_pred)
        '''
        print('y_true')
        print(np.unique(y_true))
        print(y_true.shape)
        print('y_pred')
        print(np.unique(y_pred))
        print(y_pred.shape)
        '''
        #print(np.unique(y_true==y_pred))
        #if(i>=727):
        #    print(i)
        #    print(y_pred[0,:])
        #    print(y_true[0,:])
        y_true = y_true.flatten().tolist()
        y_pred = y_pred.flatten().tolist()
        
        #tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,60,255]).ravel()
        #print(confusion_matrix(y_true, y_pred, labels=[0,60,255]).ravel())
        jaccard_scores = metrics_jaccard(y_true,y_pred,jaccard_scores)
        f1_scores = metrics_f1(y_true,y_pred,f1_scores)
        '''
        print('f1_score micro:')
        print(f1_score(y_true, y_pred,average='micro'))
        print('f1_score macro:')
        print(f1_score(y_true, y_pred,average='macro'))
        print('f1_score weighted:')
        print(f1_score(y_true, y_pred,average='weighted'))
        '''
        num+=1
        if(num%50 == 0):
            print(num)
        #    break
    print(num)
    print('Jaccard scores:')
    print(jaccard_scores)
    print('Jaccard Macro:')
    print(jaccard_scores[0]/num)
    print('Jaccard Weighted:')
    print(jaccard_scores[1]/num)
    print('F1 scores:')
    print(f1_scores)
    print('F1 Macro:')
    print(f1_scores[0]/num)
    print('F1 Weighted:')
    print(f1_scores[1]/num)

TN = FP = FN = TP = 0
epochs = 150
person = 'Keanu'
print('Number of files the training set was trained on {}'.format(2400))
print('Number of epochs for training is {}'.format(epochs))
#predictions = '300_epochs_predictions/4_epochs_predictions/' 
predictions = '650_epochs_predictions/' 
print(predictions)
gt_folder = 'test_partition_ground_truth_visible/'
files = [int(f.split('.')[0]) for f in listdir(predictions) if isfile(join(predictions, f))]
files.sort()
print(files)
read_files(files,predictions,gt_folder)
