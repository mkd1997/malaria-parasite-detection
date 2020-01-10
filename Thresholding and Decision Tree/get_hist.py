import pandas as pd
import numpy as np
import cv2
import os
import ntpath
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import sys


def get_hist(img_filename, fname_list, black_stain_list):

    img = cv2.imread(img_filename)

    img[np.where((img == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    # smoothing
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    gray_img = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    ret, th1 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

    (n, bins, pathces) = plt.hist(th1.ravel(), bins=255)
    just_fname = ntpath.basename(img_filename)
    if(n[0] > 0.0):
        fname_list.append(just_fname)
        black_stain_list.append(True)
    else:
        fname_list.append(just_fname)
        black_stain_list.append(False)

if __name__ == "__main__":    
    dir_name = sys.argv[1]
    fname_list = []
    black_stain_list = []
    label = []

    for image in os.listdir('./' + dir_name + '/'):        
        if '.png' in image:            
            img_filename = './' + dir_name + '/' + image
            get_hist(img_filename, fname_list, black_stain_list)
    label = [1] * len(fname_list)

    df_dict = {'Name': fname_list, 'black_stain': black_stain_list, 'Label': label}

    feat_df = pd.DataFrame(df_dict)
    csv_name = './features/' + dir_name + '.csv'
    feat_df.to_csv(csv_name, index=False)