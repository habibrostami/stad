import tensorflow as tf
from tensorflow.keras import layers, Model, utils, applications
import numpy as np
from keras.datasets import mnist as mnist
import pydot
import graphviz as gv
import random
import os
from cnn_rnn_settings import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELES, action_shape, state_shape

from tensorflow.python.keras.models import load_model
from micronet import micronet
from micronet import micronet
from keras.layers import Flatten, BatchNormalization


def get_resnet50_model(state_shape, action_shape):
    cnn = tf.keras.applications.resnet.ResNet50(include_top=False,
                                                weights=None,
                                                input_tensor=None,
                                                input_shape=state_shape,
                                                # <----- input shape for cnn is just the image
                                                pooling=None, classes=action_shape)
    return cnn

def get_resnet50_with_top(state_shape, action_shape):
    cnn = tf.keras.applications.resnet.ResNet50(include_top=True,
                                                weights=None,
                                                input_tensor=None,
                                                input_shape=state_shape,
                                                # <----- input shape for cnn is just the image
                                                pooling=None, classes=action_shape)
    return cnn

def get_pretrained_model(path, include_top=False, freez_layers_no=0):
    model = load_model(path)
    if include_top == False:
        pass
        model._layers.pop()
        model.summary()

    if freez_layers_no < 0:
        no_layers = len(model.layers)
    else:
        no_layers = freez_layers_no

    for layer in model.layers[:no_layers]:
        print(layer.name)
        if isinstance(layer, BatchNormalization):
            print('----------batch normalizzation')
            layer.trainable = True
        else:
            layer.trainable = False
    model.summary()
    return model


def get_cnn_model(state_shape, action_shape, qlearning=False, include_top=False):
    return micronet(state_shape, action_shape, qlearning=False, include_top=False)



def get_full_cnn_rnn_resnet50():
    cnn = get_resnet50_with_top(state_shape, action_shape)
    inp = layers.Input((None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELES), ragged=True)

    x = layers.TimeDistributed(cnn)(inp)
    #x = layers.LSTM(32, activation='relu', recurrent_dropout=0.5, dropout=0.5, return_sequences=True)(x)
    x = layers.LSTM(64, activation='relu', recurrent_dropout=0.25, dropout=0.15, return_sequences=True)(x)
    x = layers.LSTM(32, activation='relu', recurrent_dropout=0.25, dropout=0.15)(x)
    x = layers.Dense(16, activation='relu')(x)
    out = layers.Dense(action_shape, activation='softmax')(x)

    model = Model(inp, out)

    return model
def get_cnn_rnn_resnet50():
    cnn = get_resnet50_model(state_shape, action_shape)
    inp = layers.Input((None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELES), ragged=True)

    x = layers.TimeDistributed(cnn)(inp)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.TimeDistributed (layers.Dropout (0.5)) (x)
    x = layers.LSTM(32, activation='relu', recurrent_dropout=0.15, dropout=0.15, return_sequences=True)(x)
    x = layers.LSTM(16, activation='relu', recurrent_dropout=0.15, dropout=0.15, return_sequences=True)(x)
    x = layers.LSTM(16, activation='relu', recurrent_dropout=0.15, dropout=0.15)(x)
    x = layers.Dense(16, activation='relu')(x)
    out = layers.Dense(action_shape, activation='softmax')(x)

    model = Model(inp, out)

    return model
