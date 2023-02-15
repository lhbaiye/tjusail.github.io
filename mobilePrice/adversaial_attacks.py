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
parser.add_argument('--version', default='1', type=int, help='version')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


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

    def generate(self, x, y, randRate=1):
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
        if self.isRand:
            x = x + np.random.uniform(-self.ep * randRate, self.ep * randRate, x.shape)
            x = np.clip(x, self.clip_min, self.clip_max)

        x_adv = tf.Variable(x, dtype='float32')
        for i in range(self.epochs):
            with tf.GradientTape() as tape:
                loss = tf.keras.losses.categorical_crossentropy(target, self.model(x_adv))
                grads = tape.gradient(loss, x_adv)
            delta = tf.sign(grads)
            x_adv.assign_add(self.step * delta)
            x_adv = tf.clip_by_value(x_adv, clip_value_min=self.clip_min, clip_value_max=self.clip_max)
            x_adv = tf.Variable(x_adv)

        success_idx = np.where(np.argmax(self.model(x_adv), axis=1) != np.argmax(y, axis=1))[0]
        print("SUCCESS:", len(success_idx))
        return x_adv.numpy()[success_idx], y[success_idx]

if __name__ == '__main__':
    opt = parser.parse_args()

    # load the victim model
    model = load_model(version=opt.version)
    model.summary()
    X_train, X_test, X_val, y_train, y_test, y_val = get_data(version=opt.version)
    pgd = PGD(model, ep=opt.ep, epochs=opt.iters)
    advx, advy = pgd.generate(X_train, y_train)




