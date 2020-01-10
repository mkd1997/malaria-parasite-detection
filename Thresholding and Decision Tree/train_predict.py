import pandas as pd
import numpy as np
import cv2
import os
import ntpath
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

def train(x_train, y_train):
    clf = tree.DecisionTreeClassifier()

    clf = clf.fit(x_train, y_train)
    return clf


def predict(clf, x_test):
    pred = clf.predict(x_test)
    return pred

if __name__ == "__main__":
    cols = ['Name', 'black_stain', 'Label']
    feat_cols = ['black_stain']
    target = ['Label']
    train_df = pd.DataFrame(columns=cols)

    for feat_file in os.listdir('./train/Parasitized/features/'):
        if '.csv' in feat_file:
            dir_feat_df = pd.read_csv('./train/Parasitized/features/' + feat_file, dtype={'Name':str, 'black_stain':bool, 'Label':int})
            train_df = train_df.append(dir_feat_df)

    for feat_file in os.listdir('./train/Uninfected/features/'):
        if '.csv' in feat_file:
            dir_feat_df = pd.read_csv('./train/Uninfected/features/' + feat_file, dtype={'Name':str, 'black_stain':bool, 'Label':int})
            train_df = train_df.append(dir_feat_df)
    
    train_df.Name = train_df.Name.astype(str)
    train_df.black_stain = train_df.black_stain.astype(bool)
    train_df.Label = train_df.Label.astype(int)
    print(train_df.head(n = 10))
    print(train_df.info())
    x_train, x_test, y_train, y_test = tts(train_df[['Name', 'black_stain']], train_df[target], test_size=0.3)
    trained_model = train(x_train[feat_cols], y_train)



    predictions = predict(trained_model, x_test[feat_cols])
    score = accuracy_score(y_test, predictions)
    temp = pd.DataFrame()
    temp['Name'] = x_test['Name']
    temp['predictions'] = predictions
    temp['truth_values'] = y_test
    
    temp = temp[temp['predictions'] != temp['truth_values']]
    print(temp.head())
    temp.to_csv('wrong_predictions.csv', index=False)
    print('train score: ', score)

    test_df = pd.DataFrame(columns=['Name', 'black_stain'])
    
    for feat_file in os.listdir('./test/features/'):
        if '.csv' in feat_file:
            dir_feat_df = pd.read_csv('./test/features/' + feat_file, dtype={'Name':str, 'black_stain':bool})
            test_df = test_df.append(dir_feat_df)

    test_predictions = predict(trained_model, test_df[feat_cols])
    test_df.drop(columns=['black_stain'], inplace=True)
    test_df['Label'] = test_predictions
    test_df.to_csv('sub_10.csv', index=False)
