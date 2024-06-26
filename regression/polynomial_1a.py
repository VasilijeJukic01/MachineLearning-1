# -*- coding: utf-8 -*-
"""Polynomial_1a.ipynb

Automatically generated by Colaboratory.

"""

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv('bottle.csv', usecols=["Salnty", "T_degC"])

sns.scatterplot(x='Salnty', y='T_degC', data=data)
plt.title('Data Analysis')
plt.xlabel('Salinity (Salnty)')
plt.ylabel('Temperature (T_degC)')
plt.show()

data.info()

data.dropna(inplace=True)

def plot(data, w, b, h, xs, features, xlabel='x', ylabel='y', clr='r', label=None):

    def create_feature_matrix(x, features):
        features_list = []
        for deg in range(1, features+1):
            features_list.append(np.power(x, deg))
        return np.column_stack(features_list)

    print(f'W = {w.numpy()}, bias = {b.numpy()}')

    sns.lineplot(x=xs[:, 0], y=h.numpy().flatten(), color=clr, label=label)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    sns.set(style="whitegrid")
    sns.despine()

def polynomial(x, y, features, alpha=0.001, epochs=100, normalize=False, shuffle=True, regularization=None, lambda_param=0.01):
    data = {'x': x, 'y': y}
    samples = data['x'].shape[0]

    def create_feature_matrix(x, features):
        features_list = []
        for deg in range(1, features+1):
          features_list.append(np.power(x, deg))
        return np.column_stack(features_list)

    if shuffle:
        indices = np.random.permutation(data['x'].shape[0])
        data['x'], data['y'] = data['x'][indices], data['y'][indices]

    if normalize:
        data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)
        data['y'] = (data['y'] - np.mean(data['y'])) / np.std(data['y'])

    data['x'] = create_feature_matrix(data['x'], features)

    w = tf.Variable(tf.zeros(features))
    b = tf.Variable(0.0)

    def pred(x, w, b):
        w_col = tf.reshape(w, (features, 1))
        h = tf.add(tf.matmul(x, w_col), b)
        return h

    def loss(x, y, w, b):
        prediction = pred(x, w, b)
        y_col = tf.reshape(y, (-1, 1))
        mse = tf.reduce_mean(tf.square(prediction - y_col))

        if regularization == 'l1':
            lasso = lambda_param * tf.reduce_mean(tf.abs(w))
            loss = tf.add(mse, lasso)
        elif regularization == 'l2':
            ridge = lambda_param * tf.reduce_mean(tf.square(w))
            loss = tf.add(mse, ridge)
        else:
            loss = mse

        return loss

    def calculate_gradient(x, y, w, b):
        with tf.GradientTape() as tape:
            loss_val = loss(x, y, w, b)

        w_grad, b_grad = tape.gradient(loss_val, [w, b])
        return w_grad, b_grad, loss_val

    adam = tf.keras.optimizers.Adam(learning_rate=alpha)
    avg_loss = 0

    def train_step(x, y, w, b):
        w_grad, b_grad, loss = calculate_gradient(x, y, w, b)
        adam.apply_gradients(zip([w_grad, b_grad], [w, b]))
        return loss

    for epoch in range(epochs):

        epoch_loss = 0
        for sample in range(samples):
            x = data['x'][sample].reshape((1, features))
            y = data['y'][sample]

            curr_loss = train_step(x, y, w, b)
            epoch_loss += curr_loss

        epoch_loss /= samples
        avg_loss += epoch_loss
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1}/{epochs}| Avg loss: {epoch_loss:.5f}')

    avg_loss /= epochs
    xs = create_feature_matrix(np.linspace(-2, 4, 100, dtype='float32'), features)
    h = pred(xs, w, b)

    # plot(data, w, b, h, xs, features)

    return data, w, b, h, xs, features, avg_loss

x = data['Salnty'].values.astype('float32')
y = data['T_degC'].values.astype('float32')

flag = False
colors = ['r', 'c', 'g', 'y', 'm', 'tab:orange']
loss_list = []

for i in range(6):
    d, w, b, h, xs, f, loss = polynomial(x, y, i+1, alpha=0.001, epochs=50, normalize=True)
    loss_list.append(loss)

    if not flag:
        plt.scatter(d['x'][:, 0], d['y'], label='Data')
        plt.xlabel('x')
        plt.ylabel('y')
        flag=True

    plot(d, w, b, h, xs, f, clr=colors[i], label=f'Degree {i+1}')

loss_values = [tensor.numpy() for tensor in loss_list]

degrees = [1, 2, 3, 4, 5, 6]

sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
sns.lineplot(x=degrees, y=loss_values)
plt.title('Final Loss')

plt.xlabel('Degree')
plt.ylabel('Average Loss')

plt.show()