import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer

def data_imputation(df):
    data = df[['BIDLO','ASKHI','PRC','VOL','BID','ASK','SHROUT','OPENPRC','log_volume','MovAvofV5days',
        'Mov5Price','MovAvofV200days','KAMA','PRC_orcl']].values
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(data)
    data = imputer.transform(data)
    price = df["PRC"]
    return data,price

def data_imputation_without_feature_selection(df):
    data = df[['BIDLO','ASKHI','PRC','VOL','RET','BID', 'ASK', 'SHROUT', 'OPENPRC','RETX','daily_return',
               'log_volume','changeVolume01','changePrice01','changeVolume50','changePrice50',
               'MovAvofV5days','Mov5Price','MovAvofV200days','MovAvof50daysPrice','percentile',
               'rateofchange','sign','plus_minus','rateofchangePrice','signprice','money_flow_index',
               'AO','KAMA','rsi','stochastic','TSI','ultimate','williamsR','Price','Open','High','Low',
               'PRC_intc','PRC_orcl','PRC_infy']].values
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(data)
    data = imputer.transform(data)
    price = df["PRC"]
    return data,price

def split_data(data,price):
    train = data[:int(len(data)*0.8)]
    test = data[int(len(data)*0.8):]
    train_y = price[:int(len(data)*0.8)]
    test_y = price[int(len(data)*0.8):]
    return train,test,train_y,test_y

def data_transform_2d_pca(train,train_y,window_size = 60):
    X_train = []
    y_train = []
    for i in range(window_size,train.shape[0]):
        X_train.append(train[i-window_size:i,:])
        y_train.append(train_y[i])
    X_train, y_train = np.array(X_train), np.array(y_train)
    return X_train,y_train

def data_transform(train,train_y,window_size = 60):
    X_train = []
    y_train = []
    for i in range(window_size,train.shape[0]):
        X_train.append(train[i-window_size:i,:].reshape(-1,1))
        y_train.append(train_y[i])
    X_train, y_train = np.array(X_train), np.array(y_train)
    return X_train,y_train

def PCA_transform(train,out_put_size = 10):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=out_put_size)
    pca.fit(train.reshape(train.shape[0],train.shape[1]))
    train = pca.transform(train.reshape(train.shape[0],train.shape[1]))
    return train

def normalize(X_train,y_train):
    sc = MinMaxScaler(feature_range=(-1,1))
    transform_train = sc.fit_transform(X_train.reshape(X_train.shape[0],X_train.shape[1]))
    transform_train_y = sc.fit_transform(np.array(y_train).reshape(-1,1))
    return transform_train,transform_train_y,sc

def split(data, rate):
    return data.iloc[:int(len(data)*rate)],data.iloc[int(len(data)*rate):]