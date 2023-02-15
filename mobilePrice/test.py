# @Time    : 2023/2/15 15:45
# @Author  : chenxuyang
# @File    : test.py
# @Description :
# @Time    : 2023/2/13 18:42
# @Author  : chenxuyang
# @File    : main.py
# @Description : Mobile price Classification
import numpy as np
import pandas as pd

import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

# 标准化数据
def standard_data(X):
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    return sc, X_scaled

def get_data(version=1, drop_col_number=1):
    path_list = dict()
    for dirname, _, filenames in os.walk('D:/code/regression/dataset/mobilePrice'):
        for filename in filenames:
            filename, file_type = os.path.join(dirname, filename), filename.split('.')[0]
            path_list[file_type] = filename
    # 训练集
    train_df = pd.read_csv(path_list['train'])
    col2idx = {str(item): idx for idx, item in enumerate(train_df.columns)}
    idx2col = {idx: str(item) for idx, item in enumerate(train_df.columns)}
    sort_corr.drop('price_range', inplace=True)
    drop_col_id = np.random.randint(0, len(sort_corr), drop_col_number)
    drop_col_name = [idx2col[idx] for idx in drop_col_id]

    if version == 1:
        X = train_df.drop(['price_range'], axis=1)
        X = X.drop(drop_col_name, axis=1)
    else:
        X = train_df.drop(['price_range'], axis=1)
    y = train_df['price_range']
    sc, X_scaled = standard_data(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
    return X_train, X_test, X_val, y_train, y_test, y_val

def load_model(version=1, train=False):
    model = Sequential()
    filepath = "./model/weights.best.hdf5"
    if version == 1:
        model.add(Dense(18, activation='relu', input_dim=18))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(4, activation='softmax'))
        filepath = "./model/weights.best.hdf5"

    else:
        model.add(Dense(20, activation='relu', input_dim=20))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(4, activation='softmax'))
        filepath = "./model/new_weights.best.hdf5"

    X_train, X_test, X_val, y_train, y_test, y_val = get_data(version=version)
    if train:
        model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=100, callbacks=callbacks_list, verbose=0)
    else:
        model.load_weights(filepath)
        model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])


    scores = model.evaluate(X_test, y_test, verbose=0)
    print("{0}: {1:.2f}%".format(model.metrics_names[1], scores[1]*100))
    return model




def main():
    train = False
    # 采集数据
    path_list = dict()
    for dirname, _, filenames in os.walk('D:/code/regression/dataset/mobilePrice'):
        for filename in filenames:
            filename, file_type = os.path.join(dirname, filename), filename.split('.')[0]
            path_list[file_type] = filename
    # 训练集
    train_df = pd.read_csv(path_list['train'])
    X = train_df.drop(['price_range', 'px_width', 'px_height'], axis=1)
    y = train_df['price_range']
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
    ## 定义模型
    model = Sequential()
    model.add(Dense(18, activation='relu', input_dim=18))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.summary()
    filepath = "./model/weights.best.hdf5"

    if train:
        model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=100, callbacks=callbacks_list, verbose=0)
    else:
        model.load_weights(filepath)
        model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])


    scores = model.evaluate(X_test, y_test, verbose=0)
    print("{0}: {1:.2f}%".format(model.metrics_names[1], scores[1]*100))


def add_column():
    train = True
    # 采集数据
    path_list = dict()
    for dirname, _, filenames in os.walk('D:/code/regression/dataset/mobilePrice'):
        for filename in filenames:
            filename, file_type = os.path.join(dirname, filename), filename.split('.')[0]
            path_list[file_type] = filename
    # 训练集
    train_df = pd.read_csv(path_list['train'])
    X = train_df.drop(['price_range'], axis=1)
    y = train_df['price_range']
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
    ## 定义模型
    model = Sequential()
    model.add(Dense(20, activation='relu', input_dim=20))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.summary()
    filepath = "./model/new_weights.best.hdf5"

    if train:
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=100,
                            callbacks=callbacks_list, verbose=0)
    else:
        model.load_weights(filepath)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("{0}: {1:.2f}%".format(model.metrics_names[1], scores[1] * 100))
    return model



if __name__ == '__main__':
    add_column()