# @Time    : 2023/2/14 21:27
# @Author  : chenxuyang
# @File    : adversaial_attacks.py
# @Description :

import argparse
import tensorflow as tf
from main import load_model, get_data
import numpy as np
import os
import time
import pandas as pd
parser = argparse.ArgumentParser(description='black-box test case generation process')
parser.add_argument('--model', required=False, default='D:/code/DeepJudge/train_models/mnist/models/mnist_lenet5_20.h5',
                    type=str, help='victim model path')
parser.add_argument('--seeds', required=False, default='D:/code/DeepJudge/DeepJudge/seeds/mnist_max_1000seeds.npz',
                    type=str, help='selected seeds path')
parser.add_argument('--method', default='cw', type=str, help='adversarial attacks. choice: fgsm/pgd/cw')
parser.add_argument('--ep', default=0.1, type=float, help='for fgsm/pgd attack (perturbation bound)')
parser.add_argument('--iters', default=10, type=int, help='for pgd attack')
parser.add_argument('--confidence', default=5, type=float, help='for cw attack')
parser.add_argument('--cmin', default=0, type=float, help='clip lower bound')
parser.add_argument('--cmax', default=1, type=float, help='clip upper bound')
parser.add_argument('--output', default='./testcases', type=str, help='test cases saved dir')
parser.add_argument("--version", type=int, default=1, help="version")
parser.add_argument("--drop_col_number", type=int, default=2,
                    help="Number of deleted columns, if version ==0, this parameter is invalid")
parser.add_argument("--drop_col_name", type=str, default=None,
                    help="Deletes the column name of the model, If you specify the number, you don't need to specify the column name")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class statistical_result:
    drop_col_name = None
    drop_col_number = None
    version = None
    accuracy = None
    loss = None
    filepath = None
    train_attack_success_rate = None
    test_attack_success_rate = None
    train_column_attack_rate = None
    test_column_attack_rate = None
    def __init__(self, ):
        super().__init__()

    def keys(self):
        '''当对实例化对象使用dict(obj)的时候, 会调用这个方法,这里定义了字典的键, 其对应的值将以obj['name']的形式取,
           但是对象是不可以以这种方式取值的, 为了支持这种取值, 可以为类增加一个方法'''
        return ['drop_col_name', 'drop_col_number', 'version', 'accuracy', 'loss', 'filepath', 'train_attack_success_rate', 'test_attack_success_rate', 'train_column_attack_rate', 'test_column_attack_rate']

    def __getitem__(self, item):
        '''内置方法, 当使用obj['name']的形式的时候, 将调用这个方法, 这里返回的结果就是值'''
        return getattr(self, item)

class PGD:
    def __init__(self, model, ep=0.3, epochs=10, step=0.03, isRand=True, clip_min=0, clip_max=1):
        """
        args:
            model: victim model
            ep: PGD perturbation bound
            epochs: PGD iterations
            isRand: whether adding a random noise
            clip_min: clip lower bound
            clip_max: clip upper bound
        """
        self.isRand = isRand
        self.model = model
        self.ep = ep
        self.epochs = epochs
        self.step = step
        self.clip_min = clip_min
        self.clip_max = clip_max



    def generate(self, x, y, col2idx, idx2col, type='all', randRate=1, column = None):
        """
        args:
            x: normal inputs
            y: ground-truth labels
            randRate: the size of random noise

        returns:
            successed adversarial examples and corresponding ground-truth labels
        """
        # 扩充维度
        y = tf.keras.utils.to_categorical(y, 4)
        target = tf.constant(y, dtype='float32')
        # if self.isRand:
        #     x = x + np.random.uniform(-self.ep * randRate, self.ep * randRate, x.shape)
        #     x = np.clip(x, self.clip_min, self.clip_max)
        if column is not None:
            drop_col_name = column.split('-')
            drop_col_idx = [col2idx[col] for col in drop_col_name]
        else:
            drop_col_idx = []
        x_adv = tf.Variable(x, dtype='float32')
        for i in range(self.epochs):
            with tf.GradientTape() as tape:
                loss = tf.keras.losses.categorical_crossentropy(target, self.model(x_adv))
                grads = tape.gradient(loss, x_adv)
            delta = tf.sign(grads)
            if type == 'all':
                x_adv.assign_add(self.step * delta)
            else:
                delta = delta.numpy()
                for col in range(delta.shape[1]):
                    if col in drop_col_idx:
                        continue
                    delta[:, col] = 0
                delta = tf.convert_to_tensor(delta)
                x_adv.assign_add(self.step * delta)
            x_adv = tf.clip_by_value(x_adv, clip_value_min=self.clip_min, clip_value_max=self.clip_max)
            x_adv = tf.Variable(x_adv)

        success_idx = np.where(np.argmax(self.model(x_adv), axis=1) != np.argmax(y, axis=1))[0]
        return "{:.2f}%".format(len(success_idx) / len(x)  * 100)
        # return x_adv.numpy()[success_idx], y[success_idx]

if __name__ == '__main__':
    path_list = dict()
    for dirname, _, filenames in os.walk('D:/code/regression/dataset/mobilePrice'):
        for filename in filenames:
            filename, file_type = os.path.join(dirname, filename), filename.split('.')[0]
            path_list[file_type] = filename
    # 训练集
    train_df = pd.read_csv(path_list['train'])
    col2idx = {str(item): idx for idx, item in enumerate(train_df.columns)}
    idx2col = {idx: str(item) for idx, item in enumerate(train_df.columns)}

    opt = parser.parse_args()
    all_cla = dict()
    result = pd.read_csv('./result/model_acc_result.csv')

    base_model_data = result.loc[result['version'] == 0]

    drop_col_name = base_model_data['drop_col_name']
    drop_col_number = base_model_data['drop_col_number']
    # 编译模型
    base_model_model = load_model(version=base_model_data['version'].values[0]
                                  , drop_col_number=drop_col_number)
    base_model_model.load_weights(base_model_data['filepath'].values[0])
    base_model_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    base_X_train = np.load('./data/0/X_train.npy')
    base_X_test = np.load('./data/0/X_test.npy')
    base_X_val = np.load('./data/0/X_val.npy')
    base_y_train = np.load('./data/0/y_train.npy')
    base_y_test = np.load('./data/0/y_test.npy')
    base_y_val = np.load('./data/0/y_val.npy')


    for index, row in result.iterrows():
        # 封装数据
        s_r_c = statistical_result()
        s_r_c.drop_col_name = row['drop_col_name']
        s_r_c.drop_col_number = row['drop_col_number']
        s_r_c.version = row['version']
        s_r_c.accuracy = row['accuracy']
        s_r_c.loss = row['loss']
        s_r_c.filepath = row['filepath']
        all_cla[str(row['version']) + '-' +  row['drop_col_name']] = s_r_c
        # 获取删除的名称和类别
        drop_col_name = row['drop_col_name']
        drop_col_number = row['drop_col_number']
        # 编译模型
        model = load_model(version=s_r_c.version, drop_col_number=drop_col_number)
        model.load_weights(s_r_c.filepath)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        if s_r_c.version == 0:
            X_train = np.load('./data/0/X_train.npy')
            X_test = np.load('./data/0/X_test.npy')
            X_val = np.load('./data/0/X_val.npy')
            y_train = np.load('./data/0/y_train.npy')
            y_test = np.load('./data/0/y_test.npy')
            y_val = np.load('./data/0/y_val.npy')
        else:
            save_data_path = './data/{}-{}'.format(drop_col_number, drop_col_name)
            X_train = np.load('{}/X_train.npy'.format(save_data_path))
            X_test = np.load('{}/X_test.npy'.format(save_data_path))
            X_val = np.load('{}/X_val.npy'.format(save_data_path))
            y_train = np.load('{}/y_train.npy'.format(save_data_path))
            y_test = np.load('{}/y_test.npy'.format(save_data_path))
            y_val = np.load('{}/y_val.npy'.format(save_data_path))

        # 生成对抗样本
        pgd = PGD(model, ep=opt.ep, epochs=opt.iters)
        train_attack_success_rate = pgd.generate(X_train, y_train, col2idx, idx2col, type='all')

        pgd = PGD(model, ep=opt.ep, epochs=opt.iters)
        test_attack_success_rate = pgd.generate(X_test, y_test, col2idx, idx2col, type='all')

        train_column_attack_rate = 0
        test_column_attack_rate = 0

        if s_r_c.version != 0:
            pgd = PGD(base_model_model, ep=opt.ep, epochs=opt.iters)
            train_column_attack_rate = pgd.generate(base_X_train, base_y_train, col2idx, idx2col, type='add', column = s_r_c.drop_col_name)

            pgd = PGD(base_model_model, ep=opt.ep, epochs=opt.iters)
            test_column_attack_rate = pgd.generate(base_X_test, base_y_test, col2idx, idx2col, type='add', column = s_r_c.drop_col_name)

        s_r_c.train_attack_success_rate = train_attack_success_rate
        s_r_c.test_attack_success_rate = test_attack_success_rate
        s_r_c.train_column_attack_rate = train_column_attack_rate
        s_r_c.test_column_attack_rate = test_column_attack_rate

        save_file_path = './result/model_attack_result.csv'
        if not os.path.exists('./result'):
            os.mkdir('./result')
        if not os.path.exists(save_file_path):
            pd.DataFrame(dict(s_r_c), index=[0]).to_csv(save_file_path, index=False)
        else:
            pd.DataFrame(dict(s_r_c), index=[0]).to_csv(save_file_path, mode='a', header=False, index=False)




