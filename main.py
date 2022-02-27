import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


def load_dataset():
    # data set url https://archive.ics.uci.edu/ml/datasets/Student+Performance
    attributes = ["G1", "G2", "G3", "studytime", "failures", "absences", "health", "freetime"]
    data = pd.read_csv('student-mat.csv', sep=";")
    data = data[attributes]
    x_set = np.array(data.drop(["G3"], 1))
    y_set = np.array(data["G3"])
    # alternative way to split dataset without using sklearn package
    # train_dataset = data.sample(frac=0.8, random_state=0)
    # test_dataset = data.drop(train_dataset.index)
    _x_train, _x_test, _y_train, _y_test = train_test_split(x_set, y_set, test_size=0.1)
    return _x_train, _x_test, _y_train, _y_test


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.show()


def build_and_compile_model(activation_neurons):
    _model = keras.Sequential([
        layers.Dense(activation_neurons, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    _model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))
    return _model


def train_and_save_model(init_layers):
    model = build_and_compile_model(init_layers)
    history = model.fit(x_train, y_train, validation_split=0.2, verbose=1, epochs=no_of_epochs)
    plot_loss(history)
    model.save('my_model')


def load_and_predict():
    # check whether the pre trained model exists
    if os.path.exists('./my_model'):
        new_model = tf.keras.models.load_model('my_model')
        predictions = new_model.predict(x_test)
        # to predict only one use np array with one feature set
        # print(new_model.predict(np.array([x_test[0]])))
        for index, x in enumerate(predictions):
            print('prediction : ', round(float(x)), ' actual value : ', y_test[index], x_test[index])
            if round(float(x)) == y_test[index] or round(float(x)) + 1 == y_test[index] or round(float(x)) - 1 == \
                    y_test[index]:
                print('got it')
            else:
                print('failed by ', y_test[index] - round(float(x)))
        # test set evaluation
        new_model.evaluate(x_test, y_test, verbose=1)
    else:
        train_and_save_model(len(x_train[0]))


if __name__ == '__main__':
    # variables declared here are global
    x_train, x_test, y_train, y_test = load_dataset()
    retrain_model = False
    no_of_epochs = 1000
    if retrain_model:
        train_and_save_model(len(x_train[0]))
    load_and_predict()
