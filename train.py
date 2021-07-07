import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import random
import sys
import os


# set seed
def seed_everything(SEED):
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)

from util.TwoDPCA import *
from util.TTwoPCA import *
from util.data_process import *
from util.model import *
from util.waivelet import *
from util.SAE import *

def get_baseline_data(df,window_size = 60):
    data, price = data_imputation_without_feature_selection(df)
    data, price, sc = normalize(data, price)
    data, price = data_transform(data, price, window_size)
    X_train, X_test, y_train, y_test = split_data(data, price)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    xtrain, xVal, ytrain, yVal = split_data(X_train, y_train)
    return X_train, xVal, y_train, yVal,sc,X_test,y_test

def get_wt_pca_data(df,output_size = 200,window_size = 60):
    data, price = data_imputation(df)
    data, price, sc = normalize(data, price)
    data = WT(data)
    data, price = data_transform(data, price, window_size=60)
    data = PCA_transform(data, output_size)
    X_train, X_test, y_train, y_test = split_data(data, price)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    xtrain, xVal, ytrain, yVal = split_data(X_train, y_train)
    return X_train, xVal, y_train, yVal,sc,X_test,y_test

def get_wt_2dpca_data(df,output_size = 11,window_size = 60):
    data, price = data_imputation(df)
    data, price, sc = normalize(data, price)
    data = WT(data)
    data, price = data_transform_2d_pca(data, price, window_size=60)
    data = twoDPCA(data, 11)
    data = data.reshape(data.shape[0], -1)
    X_train, X_test, y_train, y_test = split_data(data, price)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    xtrain, xVal, ytrain, yVal = split_data(X_train, y_train)
    return X_train, xVal, y_train, yVal,sc,X_test,y_test

def get_wt_2d2dpca_data(df,row=16,col=10,window_size = 60):
    data, price = data_imputation(df)
    data, price, sc = normalize(data, price)
    data = WT(data)
    data, price = data_transform_2d_pca(data, price, window_size=60)
    data = ttwoDPCA(data, col, row)
    data = data.reshape(data.shape[0], -1)
    X_train, X_test, y_train, y_test = split_data(data, price)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    xtrain, xVal, ytrain, yVal = split_data(X_train, y_train)
    return X_train, xVal, y_train, yVal,sc,X_test,y_test

def get_wt_SAE_data(df,output_dim = 121,epoch=100,batch_size=256):
    data, price = data_imputation(df)
    data, price, sc = normalize(data, price)
    data = WT(data)
    data, price = data_transform(data, price, window_size=60)
    data = SAE(data)
    np.save("./data/SAE_data.npy",data)
    X_train, X_test, y_train, y_test = split_data(data, price)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    xtrain, xVal, ytrain, yVal = split_data(X_train, y_train)
    return X_train, xVal, y_train, yVal,sc,X_test,y_test

def load_SAE_data(df,window_size = 60):
    data, price = data_imputation(df)
    data, price, sc = normalize(data, price)
    data = np.load("./data/SAE_data.npy")
    X_train, X_test, y_train, y_test = split_data(data, price[window_size:])
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    xtrain, xVal, ytrain, yVal = split_data(X_train, y_train)
    return X_train, xVal, y_train, yVal, sc, X_test, y_test




if __name__ == "__main__":
    seed_everything(6000)
    df = pd.read_csv("./data/addedfeatures.csv")
    new_date = pd.to_datetime(df["date"], format='%Y%m%d', errors='ignore')
    df["date"] = new_date
    df.index = df.date
    price = df["PRC"]
    #BaseLine
    xtrain, xVal, ytrain, yVal, sc, X_test, y_test = get_baseline_data(df)
    train_LSTM_model(xtrain, ytrain, xVal, yVal, filepath="./model_weights/BASE_LSTM.h5")
    train_RNN_model(xtrain, ytrain, xVal, yVal, filepath="./model_weights/BASE_RNN.h5")
    train_GRU_model(xtrain, ytrain, xVal, yVal, filepath="./model_weights/BASE_GRU.h5")

    # SAE
    xtrain, xVal, ytrain, yVal,sc,X_test,y_test = get_wt_SAE_data(df)
    train_LSTM_model(xtrain, ytrain, xVal, yVal, filepath = "./model_weights/AE_LSTM.h5")
    train_RNN_model(xtrain, ytrain, xVal, yVal, filepath="./model_weights/AE_RNN.h5")
    train_GRU_model(xtrain, ytrain, xVal, yVal, filepath="./model_weights/AE_GRU.h5")

    #PCA
    xtrain, xVal, ytrain, yVal,sc,X_test,y_test = get_wt_pca_data(df)
    train_LSTM_model(xtrain, ytrain, xVal, yVal, filepath="./model_weights/PCA_LSTM.h5")
    train_RNN_model(xtrain, ytrain, xVal, yVal, filepath="./model_weights/PCA_RNN.h5")
    train_GRU_model(xtrain, ytrain, xVal, yVal, filepath="./model_weights/PCA_GRU.h5")

    #2DPCA
    xtrain, xVal, ytrain, yVal,sc,X_test,y_test = get_wt_2dpca_data(df)
    train_LSTM_model(xtrain, ytrain, xVal, yVal, filepath="./model_weights/2dPCA_LSTM.h5")
    train_RNN_model(xtrain, ytrain, xVal, yVal, filepath="./model_weights/2dPCA_RNN.h5")
    train_GRU_model(xtrain, ytrain, xVal, yVal, filepath="./model_weights/2dPCA_GRU.h5")

    #2DPCA
    xtrain, xVal, ytrain, yVal,sc,X_test,y_test = get_wt_2d2dpca_data(df)
    train_LSTM_model(xtrain, ytrain, xVal, yVal, filepath="./model_weights/2d2dPCA_LSTM.h5")
    train_RNN_model(xtrain, ytrain, xVal, yVal, filepath="./model_weights/2d2dPCA_RNN.h5")
    train_GRU_model(xtrain, ytrain, xVal, yVal, filepath="./model_weights/2d2dPCA_GRU.h5")




