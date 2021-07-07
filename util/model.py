import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, SimpleRNN
from tensorflow.keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error
import random
from tensorflow.keras.callbacks import ModelCheckpoint

def get_LSTM_model(X_train, y_train,xVal, yVal):
    regressor = Sequential()

    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(1, X_train.shape[2])))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.5))

    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.5))

    regressor.add(Dense(units=1))
    regressor.compile(optimizer='rmsprop', loss='mean_squared_error')
    return regressor

def get_RNN_model(X_train, y_train,xVal, yVal):
    regressor = Sequential()

    regressor.add(SimpleRNN(units=50, return_sequences=True, input_shape=(1, X_train.shape[2])))
    regressor.add(Dropout(0.2))

    regressor.add(SimpleRNN(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(SimpleRNN(units=50, return_sequences=True))
    regressor.add(Dropout(0.5))

    regressor.add(SimpleRNN(units=50))
    regressor.add(Dropout(0.5))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='rmsprop', loss='mean_squared_error')
    return regressor

def get_GRU_model(X_train, y_train,xVal, yVal):
    regressor = Sequential()

    regressor.add(GRU(units=50, return_sequences=True, input_shape=(1, X_train.shape[2])))
    regressor.add(Dropout(0.2))

    regressor.add(GRU(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(GRU(units=50, return_sequences=True))
    regressor.add(Dropout(0.5))

    regressor.add(GRU(units=50))
    regressor.add(Dropout(0.5))

    regressor.add(Dense(units=1))
    regressor.compile(optimizer='rmsprop', loss='mean_squared_error')
    return regressor

def train_LSTM_model(X_train, y_train,xVal, yVal,filepath):
    regressor = get_LSTM_model(X_train, y_train,xVal, yVal)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    regressor.fit(X_train, y_train, epochs=100, batch_size=300, validation_data=(xVal, yVal), callbacks=[checkpoint])

def train_RNN_model(X_train, y_train,xVal, yVal,filepath):
    regressor = get_RNN_model(X_train, y_train,xVal, yVal)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    regressor.fit(X_train, y_train, epochs=100, batch_size=300, validation_data=(xVal, yVal), callbacks=[checkpoint])

def train_GRU_model(X_train, y_train,xVal, yVal,filepath):
    regressor = get_GRU_model(X_train, y_train,xVal, yVal)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    regressor.fit(X_train, y_train, epochs=100, batch_size=300, validation_data=(xVal, yVal), callbacks=[checkpoint])