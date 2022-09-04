import pickle
import numpy as np
import cv2


def data(path):
    with open(path, mode='rb') as f:
        train = pickle.load(f)
        
    return train


def shuffle(x, y):
    num = np.random.permutation(x.shape[0])
    x = x[num]
    y = y[num]
    
    return x, y


def crop(img):
    im_c = img[4:29, 4:29, :] 
    im_c = cv2.resize(im_c, (32,32))
    
    return im_c


def adjust_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))


def rotation(image):
    rows= image.shape[0]
    cols = image.shape[1]
    
    # rotation
    angle = [15,-15]
    rotatie = cv2.getRotationMatrix2D((cols/2,rows/2),np.random.choice(angle),1)
    img_rot = cv2.warpAffine(image,rotatie,(cols,rows))
    
    # crop
    img_crop = crop(img_rot)
    
    return img_crop


def to_one_hot(Y_n):
    Y_new = np.zeros([len(Y_n), 43])
    for i in range(Y_new.shape[0]):
        Y_new[i, Y_n[i]] = 1
    return Y_new

def classes(Y_n):
    Y_nou = np.zeros(len(Y_n))
    for i in range(len(Y_nou)):
        Y_nou[i] = np.argmax(Y_n[i])
    return (Y_nou).astype('uint8')


def standard(img):
    return ((img - np.min(img))/(np.max(img)-np.min(img)))-0.5
