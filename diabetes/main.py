# @Time    : 2023/2/13 18:42
# @Author  : chenxuyang
# @File    : main.py
# @Description : Mobile price Classification
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
from utils import standard_data, split_data
# 标准化数据

def get_data():
    path_list = dict()
    for dirname, _, filenames in os.walk('D:/code/regression/dataset/diabetes'):
        for filename in filenames:
            filename, file_type = os.path.join(dirname, filename), filename.split('.')[0]
            path_list[file_type] = filename
    # 训练集
    data = pd.read_csv(path_list['diabetes'])
    # 数据处理
    data.loc[data['Pregnancies'] > 13, 'Pregnancies'] = np.nan
    data.loc[data['Glucose'] < 40 , 'Glucose'] = np.nan
    data.loc[data['BloodPressure'] < 40, 'BloodPressure'] = np.nan
    data.loc[data['BloodPressure'] > 120, 'BloodPressure'] = np.nan
    data.loc[data['SkinThickness'] < 12, 'SkinThickness'] = np.nan
    data.loc[data['Insulin'] == 0, 'Insulin'] = np.nan
    data.loc[data['Insulin'] > 500, 'Insulin'] = np.nan
    data.loc[data['BMI'] == 0, 'BMI'] = np.nan
    X = data.drop(columns=['Outcome'], axis=1)
    y = data['Outcome']
    imputer = KNNImputer(n_neighbors=5)
    imputer.fit(X)
    X = imputer.transform(X)

    return X, y

def load_model(version=0, drop_col_number=1):
    model = Sequential()
    if version != 0:
        model.add(Dense(8-drop_col_number, activation='relu', input_dim=8-drop_col_number))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        # 二分类问题
        model.add(Dense(1, activation='sigmoid'))

    else:
        model.add(Dense(8, activation='relu', input_dim=8))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        # 二分类问题
        model.add(Dense(1, activation='sigmoid'))

    return model


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Experiment")
    parser.add_argument("--version", type=int, default=1, help="version")
    parser.add_argument("--drop_col_number", type=int, default=2, help="Number of deleted columns, if version ==0, this parameter is invalid")
    parser.add_argument("--drop_col_name", type=str, default=None, help="Deletes the column name of the model, If you specify the number, you don't need to specify the column name")
    parser.add_argument("--train", type=bool, default=True, help="train or not")
    params = parser.parse_args()
    return params

def main(params):
    if not os.path.exists('./data/0'):
        os.makedirs('./data/0')
    if params.version == 0:
        X, y= get_data()
        sc, X_scaled = standard_data(X)
        X_train, X_test, X_val, y_train, y_test, y_val = split_data(X_scaled, y)
        np.save('./data/0/X_train.npy', X_train)
        np.save('./data/0/X_test.npy', X_test)
        np.save('./data/0/X_val.npy', X_val)
        np.save('./data/0/y_train.npy', y_train)
        np.save('./data/0/y_test.npy', y_test)
        np.save('./data/0/y_val.npy', y_val)
    else:
        X_train = np.load('./data/0/X_train.npy')
        X_test = np.load('./data/0/X_test.npy')
        X_val = np.load('./data/0/X_val.npy')
        y_train = np.load('./data/0/y_train.npy')
        y_test = np.load('./data/0/y_test.npy')
        y_val = np.load('./data/0/y_val.npy')
        print('load data from npy')

    path_list = dict()
    for dirname, _, filenames in os.walk('D:/code/regression/dataset/diabetes'):
        for filename in filenames:
            filename, file_type = os.path.join(dirname, filename), filename.split('.')[0]
            path_list[file_type] = filename
    # 训练集
    train_df = pd.read_csv(path_list['diabetes'])
    col2idx = {str(item): idx for idx, item in enumerate(train_df.columns)}
    idx2col = {idx: str(item) for idx, item in enumerate(train_df.columns)}
    if params.drop_col_name is None:
        drop_col_id = np.random.choice(range(len(train_df.columns)-1), params.drop_col_number, replace=False)
        drop_col_name = [idx2col[idx] for idx in drop_col_id]
    else:
        drop_col_name = params.drop_col_name.split('*')

    if params.version != 0:
        columns = [item for item in col2idx.keys()]
        columns = columns[:-1]

        X_train = pd.DataFrame(X_train, columns=columns)
        X_test = pd.DataFrame(X_test, columns=columns)
        X_val = pd.DataFrame(X_val, columns=columns)
        X_train = X_train.drop(drop_col_name, axis=1)
        X_test = X_test.drop(drop_col_name, axis=1)
        X_val = X_val.drop(drop_col_name, axis=1)
        # 判断是根据名称进行删除
        if params.drop_col_name is not None:
            drop_col_number = len(drop_col_name)
        else:
            drop_col_number = params.drop_col_number
        save_data_path = './data/{}-{}-{}'.format(params.version, drop_col_number, "-".join(drop_col_name))
        if not os.path.exists(save_data_path):
            os.makedirs(save_data_path)
        np.save('{}/X_train.npy'.format(save_data_path), X_train)
        np.save('{}/X_test.npy'.format(save_data_path), X_test)
        np.save('{}/X_val.npy'.format(save_data_path), X_val)
        np.save('{}/y_train.npy'.format(save_data_path), y_train)
        np.save('{}/y_test.npy'.format(save_data_path), y_test)
        np.save('{}/y_val.npy'.format(save_data_path), y_val)
        print('save data to npy')
    else:
        drop_col_name = []
        drop_col_number = 0


    model = load_model(version=params.version, drop_col_number=drop_col_number)
    model.summary()
    if os.path.exists('./model') is False:
        os.mkdir('./model')
    if params.version == 0:
        filepath = './model/{}-weights.hdf5'.format(drop_col_number)
    else:
        filepath = './model/{}-{}-{}-weights.hdf5'.format(params.version, drop_col_number, "-".join(drop_col_name))
    if params.train:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=200,
                            callbacks=callbacks_list, verbose=0)
    else:
        model.load_weights(filepath)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    scores = model.evaluate(X_test, y_test, verbose=0)
    result = {'accuracy': "{:.2f}%".format(scores[1] * 100), 'loss': scores[0],
              'drop_col_name': "-".join(drop_col_name), 'drop_col_number': drop_col_number, 'version': params.version, 'filepath': filepath}
    save_file_path = './result/model_acc_result.csv'
    if not os.path.exists('./result'):
        os.mkdir('./result')
    if not os.path.exists(save_file_path):
        pd.DataFrame(result, index=[0]).to_csv(save_file_path, index=False)
    else:
        pd.DataFrame(result, index=[0]).to_csv(save_file_path, mode='a', header=False, index=False)


def start(epoch, drop_col_number):
    version = 1
    for i in range(epoch):
        params = get_args()
        params.drop_col_number = drop_col_number
        params.version = version
        main(params)
        version += 1


if __name__ == '__main__':
    setting_list = [
        {'epoch': 8, 'drop_col_number': 1},
        {'epoch': 20, 'drop_col_number': 2},
        {'epoch': 20, 'drop_col_number': 3},
    ]

    for setting in setting_list:
        start(setting['epoch'], setting['drop_col_number'])