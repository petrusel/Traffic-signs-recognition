
import numpy as np

from util_functions import *


train = data('D:\\AI\\traffic_signs_recognizer\\train.p')
valid = data('D:\\AI\\traffic_signs_recognizer\\valid.p')
test = data('D:\\AI\\traffic_signs_recognizer\\test.p')
    

X_train, Y_train = train['features'], train['labels']
X_test, Y_test = valid['features'], valid['labels']
X_val, Y_val = test['features'], test['labels'] 


X_all = np.concatenate([X_train, X_val], axis=0) 
Y_all = np.concatenate([Y_train, Y_val], axis=0)


X_train_aug, Y_train_aug = X_all, Y_all


for i in range(0, 43):
    
    class_records = np.where(Y_train_aug==i)[0].size
    ovr_sample =  class_records
    samples = X_train_aug[np.where(Y_train_aug==i)[0]]
    X_aug = []
    Y_aug = [i] * ovr_sample
    
    for x in range(ovr_sample):
        img = samples[x % class_records]
        if np.mean(img) < 20:
            trans_img=adjust_gamma(img, 2.5)
        else:
            trans_img=rotation(img)
                
        X_aug.append(trans_img)
        
    X_train_aug = np.concatenate((X_train_aug, X_aug), axis=0)
    Y_train_aug = np.concatenate((Y_train_aug, Y_aug)) 


Y_train = to_one_hot(Y_train_aug)
Y_test = to_one_hot(Y_test)


X_train = standard(X_train_aug)
X_test = standard(X_test)


