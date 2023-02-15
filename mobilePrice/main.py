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

def get_data(version=0, drop_col_number=1, drop_col_name=None):
    path_list = dict()
    for dirname, _, filenames in os.walk('D:/code/regression/dataset/mobilePrice'):
        for filename in filenames:
            filename, file_type = os.path.join(dirname, filename), filename.split('.')[0]
            path_list[file_type] = filename
    # 训练集
    train_df = pd.read_csv(path_list['train'])
    col2idx = {str(item): idx for idx, item in enumerate(train_df.columns)}
    idx2col = {idx: str(item) for idx, item in enumerate(train_df.columns)}
    # sort_corr = train_df.corr(method='spearman')['price_range'].sort_values(ascending=False)
    # sort_corr.drop('price_range', inplace=True)
    if drop_col_name is None:
        drop_col_id = np.random.randint(0, len(train_df.columns), drop_col_number)
        drop_col_name = [idx2col[idx] for idx in drop_col_id]
    else:
        drop_col_name = drop_col_name.split('*')

    if version != 0:
        X = train_df.drop(['price_range'], axis=1)
        X = X.drop(drop_col_name, axis=1)
    else:
        drop_col_name = []
        X = train_df.drop(['price_range'], axis=1)
    y = train_df['price_range']
    return X, y, col2idx, idx2col, drop_col_name

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
    return X_train, X_test, X_val, y_train, y_test, y_val

def load_model(version=0, drop_col_number=1):
    model = Sequential()
    if version != 0:
        model.add(Dense(20-drop_col_number, activation='relu', input_dim=20-drop_col_number))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(4, activation='softmax'))

    else:
        model.add(Dense(20, activation='relu', input_dim=20))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(4, activation='softmax'))

    return model


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Experiment")
    parser.add_argument("--version", type=int, default=1, help="version")
    parser.add_argument("--drop_col_number", type=int, default=3, help="Number of deleted columns, if version ==0, this parameter is invalid")
    parser.add_argument("--drop_col_name", type=str, default=None, help="Deletes the column name of the model, If you specify the number, you don't need to specify the column name")
    parser.add_argument("--train", type=bool, default=True, help="train or not")
    params = parser.parse_args()
    return params

def main():
    params = get_args()

    X, y, col2idx, idx2col, drop_col_name = get_data(version=params.version, drop_col_number=params.drop_col_number,
                                                     drop_col_name=params.drop_col_name)
    sc, X_scaled = standard_data(X)
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(X_scaled, y)

    if params.drop_col_name is not None:
        drop_col_number = len(drop_col_name)
    else:
        drop_col_number = params.drop_col_number
    if params.version == 0:
        drop_col_number = 0

    model = load_model(version=params.version, drop_col_number=drop_col_number)
    model.summary()
    if os.path.exists('./model') is False:
        os.mkdir('./model')
    if params.version == 0:
        filepath = './model/{}-weights.hdf5'.format(drop_col_number)
    else:
        filepath = './model/{}-{}-weights.hdf5'.format(drop_col_number, "-".join(drop_col_name))
    if params.train:
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
    result = {'accuracy': "{:.2f}%".format(scores[1] * 100), 'loss': scores[0],
              'drop_col_name': "-".join(drop_col_name), 'drop_col_number': drop_col_number, 'version': params.version}
    save_file_path = './result/model_acc_result.csv'
    if not os.path.exists('./result'):
        os.mkdir('./result')
    if not os.path.exists(save_file_path):
        pd.DataFrame(result, index=[0]).to_csv(save_file_path, index=False)
    else:
        pd.DataFrame(result, index=[0]).to_csv(save_file_path, mode='a', header=False, index=False)




if __name__ == '__main__':
    main()