# @Time    : 2023/2/15 18:47
# @Author  : chenxuyang
# @File    : utils.py
# @Description :
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def standard_data(X):
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    return sc, X_scaled


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
    return X_train, X_test, X_val, y_train, y_test, y_val