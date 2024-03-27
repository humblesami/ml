from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf


def plot_linear():
    x = [1, 2, 2.5, 3, 4]
    y = [1, 4, 7, 9, 15]
    plt.plot(x, y, 'ro')
    plt.axis((0, 6, 0, 20))

    plt.plot(x, y, 'ro')
    plt.axis((0, 6, 0, 20))
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))


df_train = pd.read_csv('./datasets/titanic/train.csv') # training data
df_eval = pd.read_csv('./datasets/titanic/eval.csv') # testing data
y_train = df_train.pop('survived')
y_eval = df_eval.pop('survived')

print(df_train.head())
print(df_train.describe())
print(df_train.shape)
print(y_train.head())


def plot_age():
    df_train.age.hist(bins=20)


def plot_sex():
    df_train.sex.value_counts().plot(kind='barh')


def plot_counts():
    df_train['class'].value_counts().plot(kind='barh')


def plot_survived():
    pd.concat([df_train, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')


plot_survived()
plt.show()

