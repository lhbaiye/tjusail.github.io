# @Time    : 2023/4/3 16:27
# @Author  : chenxuyang
# @File    : test.py
# @Description :
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
def get_data():
    path_list = dict()
    for dirname, _, filenames in os.walk('D:/code/pythonProject/regression/dataset/diabetes'):
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



model1 = load_model(version=1, drop_col_number=2)
model1.load_weights('./model/2-2-Insulin-Pregnancies-weights.hdf5')
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X, y = get_data()
model1.summary()
# model1层权重与名称对应
weights_map = dict()
for i in range(len(model1.layers)):
    weights_map[i] = model1.layers[i].name

model1.get_weights()
model2 = load_model(version=0, drop_col_number=0)
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
for i in range(len(model1.layers)):
    if i == 0 or i == 1:
        continue
    model2.layers[i].set_weights(model1.layers[i].get_weights())

