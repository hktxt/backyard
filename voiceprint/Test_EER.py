import numpy as np
import os
import time
import itertools
import random
from numpy import linalg as LA
def ConsinDistance(feaV1, feaV2):
        return np.dot(feaV1, feaV2) / (LA.norm(feaV1) * LA.norm(feaV2))
def calculate_eer(y, y_score):
    # y denotes groundtruth scores,
    # y_score denotes the prediction scores.
    from scipy.optimize import brentq
    from sklearn.metrics import roc_curve
    from scipy.interpolate import interp1d

    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh
import librosa
if __name__ == '__main__':

       rcg=np.load('test/rcg9.npy')
       reg=np.load('test/reg9.npy')
       reg=reg.item()
       rcg=rcg.item()
       scores=[]
       labels=[]
       for key0,value0 in rcg.items():
           id0=key0.split('/')[0]
           feat0=value0
           for key1,value1 in reg.items():
               id1=key1.split('/')[0]
               feat1=value1 
               score =ConsinDistance(feat0,feat1)  
               scores.append(score)
               label=0 
               if id1==id0:
                  label=1
               labels.append(label)

       labels=np.array(labels)
       scores=np.array(scores)

       eer, thresh = calculate_eer(labels, scores)

       print('Thresh : {}, EER: {}'.format(thresh, eer))

  
                    
                  
              
           
       



 
        








