import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import keras
from keras import Model, models, optimizers, Sequential
from keras import layers
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetMobile
from keras.applications.resnet import ResNet50,ResNet101,ResNet152
from keras.applications.densenet import DenseNet121
from micronet import simple_xception
from keras.models import load_model
from keras import regularizers

from keras.layers import Flatten, BatchNormalization, Dropout, Dense, MaxPooling2D, Conv2D

from keras.applications.inception_v3 import  InceptionV3
import tensorflow as tf

from settings import checkpoint_filepath, crc_checkpoint_filepath

from keras.applications.mobilenet import MobileNet


def freez_layers(model,no_layers):

    print('---------------------------freezing layer', len(model.layers))
    for layer in model.layers[:no_layers]:
        print(layer.name)
        if isinstance(layer, BatchNormalization):
            print('----------batch normalizzation')
            layer.trainable = True
        else:
            layer.trainable = False
    return model

def get_pretrained_Xception(state_shape, action_shape, learning_rate, qlearning = True, no_freez_layers = 0):
    model = load_model(crc_checkpoint_filepath)
    loss_function = 'mean_squared_error'
    if qlearning:
        loss_function = 'mean_squared_error'
    else:
        #loss_function = 'categorical_crossentropy'
        loss_function = 'binary_crossentropy'
    if no_freez_layers < 0:
        no_freez_layers = len(model.layers) + no_freez_layers

    model = freez_layers(model, no_freez_layers)
    model.compile(loss=loss_function, optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

    model.summary()
    return model




def  get_DenseNet121  (state_shape, action_shape, learning_rate, qlearning = True,  no_freez_layers = 0):
    print('state shape = ', state_shape)
    print('action_shape =', action_shape)
    init = tf.keras.initializers.he_uniform()

    pretrained_model = DenseNet121(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=state_shape,
        pooling=None,
    )
    if pretrained_model.output.shape.ndims > 2:
        output = Flatten()(pretrained_model.output)
        print('Flatten is called ----------------------------------------------')
    else:
        output = pretrained_model.output
        print('direct output is called ----------------------------------------------')

    output = Dense(32, activation='relu', kernel_initializer=init)(output)
    #output = BatchNormalization()(output)
    #output = Dropout(0.5)(output)

    #output = BatchNormalization()(output)
    #output = Dropout(0.5)(output)
    output = Dense(16, activation='relu', kernel_initializer=init)(output)
    #output = BatchNormalization()(output)
    #output = Dropout(0.5)(output)
    output = Dense(8, activation='relu', kernel_initializer=init)(output)
    #output = BatchNormalization()(output)
    #output = Dropout(0.5)(output)

    if qlearning:
        output = Dense(action_shape, activation='linear', kernel_initializer=init)(output)
    else:
        output = Dense(action_shape, activation='softmax', kernel_initializer=init)(output)

    model = Model(pretrained_model.input, output)

    loss_function = 'mean_squared_error'
    if qlearning:
        loss_function = 'mean_squared_error'
    else:
        loss_function = 'categorical_crossentropy'

    model = freez_layers(model, no_freez_layers)
    model.compile(loss=loss_function, optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

    model.summary()
    return model


def  get_InceptionV3  (state_shape, action_shape, learning_rate, qlearning = True,  no_freez_layers = 0):
    print('state shape = ', state_shape)
    print('action_shape =', action_shape)
    init = tf.keras.initializers.he_uniform()

    pretrained_model = InceptionV3(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=state_shape,
        pooling=None,
    )
    if pretrained_model.output.shape.ndims > 2:
        output = Flatten()(pretrained_model.output)
        print('Flatten is called ----------------------------------------------')
    else:
        output = pretrained_model.output
        print('direct output is called ----------------------------------------------')

    output = Dense(32, activation='relu', kernel_initializer=init)(output)
    #output = BatchNormalization()(output)
    #output = Dropout(0.5)(output)

    #output = BatchNormalization()(output)
    #output = Dropout(0.5)(output)
    output = Dense(16, activation='relu', kernel_initializer=init)(output)
    #output = BatchNormalization()(output)
    #output = Dropout(0.5)(output)
    output = Dense(8, activation='relu', kernel_initializer=init)(output)
    #output = BatchNormalization()(output)
    #output = Dropout(0.5)(output)

    if qlearning:
        output = Dense(action_shape, activation='linear', kernel_initializer=init)(output)
    else:
        output = Dense(action_shape, activation='softmax', kernel_initializer=init)(output)

    model = Model(pretrained_model.input, output)

    loss_function = 'mean_squared_error'
    if qlearning:
        loss_function = 'mean_squared_error'
    else:
        loss_function = 'categorical_crossentropy'

    model = freez_layers(model, no_freez_layers)
    model.compile(loss=loss_function, optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

    model.summary()
    return model



def  get_NASNetMobile(state_shape, action_shape, learning_rate, qlearning = True, no_freez_layers = 0):
    print('state shape = ', state_shape)
    print('action_shape =', action_shape)
    init = tf.keras.initializers.he_uniform()

    pretrained_model = NASNetMobile(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=state_shape,
        pooling=None,
    )

    if pretrained_model.output.shape.ndims > 2:
        output = Flatten()(pretrained_model.output)
        print('Flatten is called ----------------------------------------------')
    else:
        output = pretrained_model.output
        print('direct output is called ----------------------------------------------')

    #output = BatchNormalization()(output)
    #output = Dropout(0.5)(output)
    #output = Dense(48, activation='relu', kernel_initializer=init)(output)
    #output = BatchNormalization()(output)
    #output = Dropout(0.5)(output)
    output = Dense(24, activation='relu', kernel_initializer=init)(output)
    #output = BatchNormalization()(output)
    #output = Dropout(0.5)(output)

    output = Dense(12, activation='relu', kernel_initializer=init)(output)
    #output = BatchNormalization()(output)
    #output = Dropout(0.5)(output)

    if qlearning:
        output = Dense(action_shape, activation='linear', kernel_initializer=init)(output)
    else:
        output = Dense(action_shape, activation='softmax', kernel_initializer=init)(output)


    #pretrained_model = freez_layers(pretrained_model,no_freez_layers)

    model = Model(pretrained_model.input, output)

    loss_function = 'mean_squared_error'
    if qlearning:
        loss_function = 'mean_squared_error'
    else:
        #loss_function = 'categorical_crossentropy'
        loss_function = 'binary_crossentropy'

    model = freez_layers(model, no_freez_layers)
    model.compile(loss=loss_function, optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

    model.summary()
    return model

def  get_InceptionRes(state_shape, action_shape, learning_rate, qlearning = True, no_freez_layers = 0):
    print('state shape = ', state_shape)
    print('action_shape =', action_shape)
    init = tf.keras.initializers.he_uniform()

    pretrained_model = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=state_shape,
        pooling=None,
    )

    if pretrained_model.output.shape.ndims > 2:
        output = Flatten()(pretrained_model.output)
        print('Flatten is called ----------------------------------------------')
    else:
        output = pretrained_model.output
        print('direct output is called ----------------------------------------------')

    #output = BatchNormalization()(output)
    #output = Dropout(0.5)(output)
    #output = Dense(48, activation='relu', kernel_initializer=init)(output)
    #output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(24, activation='relu', kernel_initializer=init)(output)
    #output = BatchNormalization()(output)
    output = Dropout(0.5)(output)

    output = Dense(12, activation='relu', kernel_initializer=init)(output)
    #output = BatchNormalization()(output)
    output = Dropout(0.5)(output)

    if qlearning:
        output = Dense(action_shape, activation='linear', kernel_initializer=init)(output)
    else:
        output = Dense(action_shape, activation='softmax', kernel_initializer=init)(output)


    #pretrained_model = freez_layers(pretrained_model,no_freez_layers)

    model = Model(pretrained_model.input, output)

    loss_function = 'mean_squared_error'
    if qlearning:
        loss_function = 'mean_squared_error'
    else:
        #loss_function = 'categorical_crossentropy'
        loss_function = 'binary_crossentropy'

    model = freez_layers(model, no_freez_layers)
    model.compile(loss=loss_function, optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['AUC','accuracy'])

    model.summary()
    return model


def  get_Vgg16(state_shape, action_shape, learning_rate, qlearning = True,  no_freez_layers = 0):
    print('state shape = ', state_shape)
    print('action_shape =', action_shape)
    init = tf.keras.initializers.he_uniform()

    pretrained_model = VGG16(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=state_shape,
        pooling=None,
    )
    if pretrained_model.output.shape.ndims > 2:
        output = Flatten()(pretrained_model.output)
        print('Flatten is called ----------------------------------------------')
    else:
        output = pretrained_model.output
        print('direct output is called ----------------------------------------------')

    output = Dense(32, activation='relu', kernel_initializer=init)(output)
    #output = BatchNormalization()(output)
    #output = Dropout(0.5)(output)

    #output = BatchNormalization()(output)
    #output = Dropout(0.5)(output)
    output = Dense(16, activation='relu', kernel_initializer=init)(output)
    ##output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(8, activation='relu', kernel_initializer=init)(output)
    #output = BatchNormalization()(output)
    #output = Dropout(0.5)(output)

    if qlearning:
        output = Dense(action_shape, activation='linear', kernel_initializer=init)(output)
    else:
        output = Dense(action_shape, activation='softmax', kernel_initializer=init)(output)

    model = Model(pretrained_model.input, output)

    loss_function = 'mean_squared_error'
    if qlearning:
        loss_function = 'mean_squared_error'
    else:
        loss_function = 'categorical_crossentropy'

    model = freez_layers(model, no_freez_layers)
    model.compile(loss=loss_function, optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

    model.summary()
    return model

def  get_simple_xception(state_shape, action_shape, learning_rate,  qlearning = True,  no_freez_layers = 0):

    model = simple_xception(state_shape, action_shape, qlearning)
    init = tf.keras.initializers.he_uniform()
    loss_function = 'mean_squared_error'
    if qlearning :
        loss_function = 'mean_squared_error'
    else:
        loss_function = 'categorical_crossentropy'

    model = freez_layers(model, no_freez_layers)
    model.compile(loss=loss_function, optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy','AUC'])

    print('++++++++++++++++++++++++++++++++++++++')
    model.summary()
    print('++++++++++++++++++++++++++++++++++++++')

    return model

def  get_dann_xception(state_shape, action_shape, learning_rate,  qlearning = True,  no_freez_layers = 0):
    pass
'''
    dann = XceptionDANN()
    model = XceptionDANN.model_label_predictor

    init = tf.keras.initializers.he_uniform()
    loss_function = 'mean_squared_error'
    if qlearning :
        loss_function = 'mean_squared_error'
    else:
        loss_function = 'categorical_crossentropy'

    model = freez_layers(model, no_freez_layers)
    model.compile(loss=loss_function, optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy','AUC'])

    print('++++++++++++++++++++++++++++++++++++++')
    model.summary()
    print('++++++++++++++++++++++++++++++++++++++')

    return model
'''

def  get_crc_simple_xception(state_shape, action_shape, learning_rate,  qlearning = True,  no_freez_layers = 0):

    model = simple_xception(state_shape, action_shape, qlearning)
    init = tf.keras.initializers.he_uniform()
    loss_function = 'mean_squared_error'
    if qlearning :
        loss_function = 'mean_squared_error'
    else:
        loss_function = 'categorical_crossentropy'

    model = freez_layers(model, no_freez_layers)
    model.compile(loss=loss_function, optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy','AUC'])

    print('++++++++++++++++++++++++++++++++++++++')
    model.summary()
    print('++++++++++++++++++++++++++++++++++++++')

    return model

def  get_Xception(state_shape, action_shape, learning_rate,  qlearning = True,  no_freez_layers = 0):
    print('state shape = ', state_shape)
    print('action_shape =', action_shape)
    init = tf.keras.initializers.he_uniform()

    pretrained_model = Xception(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=state_shape,
        pooling=None,
    )
    if pretrained_model.output.shape.ndims > 2:
        output = Flatten()(pretrained_model.output)
    else:
        output = pretrained_model.output

    #reg1 = keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    output = BatchNormalization()(output)
    output = Dropout(0.6)(output)
    output = Dense(24, activation='relu', kernel_initializer=init)(output)
    output = BatchNormalization()(output)
    output = Dropout(0.6)(output)
    output = Dense(6, activation='relu', kernel_initializer=init)(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    if qlearning :
        output = Dense(action_shape, activation='linear',kernel_initializer=init)(output)
    else:
        output = Dense(action_shape, activation='softmax', kernel_initializer=init)(output)

    model = Model(pretrained_model.input,output)

    loss_function = 'mean_squared_error'
    if qlearning :
        loss_function = 'mean_squared_error'
    else:
        loss_function = 'categorical_crossentropy'

    model = freez_layers(model, no_freez_layers)
    model.compile(loss=loss_function, optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy','AUC'])

    print('++++++++++++++++++++++++++++++++++++++')
    model.summary()
    print('++++++++++++++++++++++++++++++++++++++')

    return model


def  get_Xception_for_tta(state_shape, action_shape, learning_rate,  qlearning = True,  no_freez_layers = 0):
    print('state shape = ', state_shape)
    print('action_shape =', action_shape)
    init = tf.keras.initializers.he_uniform()

    pretrained_model = Xception(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=state_shape,
        pooling=None,
    )
    if pretrained_model.output.shape.ndims > 2:
        output = Flatten()(pretrained_model.output)
    else:
        output = pretrained_model.output

    #reg1 = keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    output = BatchNormalization()(output)
    output = Dropout(0.4)(output)
    output = Dense(48, activation='relu', kernel_initializer=init)(output)
    output = BatchNormalization()(output)
    output = Dropout(0.4)(output)
    output = Dense(24, activation='relu', kernel_initializer=init)(output)
    output = BatchNormalization()(output)
    output = Dropout(0.4)(output)
    if qlearning :
        output = Dense(action_shape, activation='linear',kernel_initializer=init)(output)
    else:
        output = Dense(action_shape, activation='softmax', kernel_initializer=init)(output)

    model = Model(pretrained_model.input,output)

    loss_function = 'mean_squared_error'
    if qlearning :
        loss_function = 'mean_squared_error'
    else:
        loss_function = 'categorical_crossentropy'

    model = freez_layers(model, no_freez_layers)
    model.compile(loss=loss_function, optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

    print('++++++++++++++++++++++++++++++++++++++')
    model.summary()
    print('++++++++++++++++++++++++++++++++++++++')

    return model


def get_simple_net(state_shape, action_shape, learning_rate, qlearning = True, no_freez_layers = 0):
    model = models.Sequential()

    # Conv. 1
    model.add(layers.Conv2D(32, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros', input_shape=state_shape))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.2))

    # Conv. 2,3 & 4
    model.add(layers.Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.2))


    # Conv. 5
    model.add(layers.Conv2D(96, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))

    # Max Pooling
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    #        self.model.add(layers.Dropout(0.2))
    
    # Conv. 6,7,8 & 9
    model.add(layers.Conv2D(96, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(96, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(96, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(96, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.2))

    # Conv. 10
    model.add(layers.Conv2D(144, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))

    # Max Pooling
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    #        self.model.add(layers.Dropout(0.3))

    # Conv. 11
    model.add(layers.Conv2D(144, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.3))

    # Conv. 12
    model.add(layers.Conv2D(178, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.3))

    # Conv. 13

    model.add(layers.Conv2D(216, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                                 bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',
                                             moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.3))


    # Global Ma Pooling
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    # Classifier
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))

    if qlearning :
        model.add(layers.Dense(action_shape, activation='linear'))

        optmz = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=optmz,
                           loss='mean_squared_error',
                           # loss='binary_crossentropy',
                           metrics=['accuracy'])
    else:
        model.add(layers.Dense(action_shape, activation='softmax'))

        optmz = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=optmz,
                           #loss='categorical_crossentropy',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    model.summary()

    return model


def get_sequence_model(state_shape, action_shape, learning_rate, qlearning = True):
    model = models.Sequential()

    g_kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)# regularizers.l2(1e-3) #None #regularizers.l1_l2(l1=1e-5, l2=1e-4),
    g_bias_regularizer=None #regularizers.l2(1e-4),
    g_activity_regularizer=regularizers.l2(1e-4)

    # Conv. 1
    model.add(layers.Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                            bias_initializer='zeros', input_shape=state_shape))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.2))

    # Conv. 2,3 & 4
    model.add(layers.Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #model.add(layers.MaxPool2D(pool_size=(2, 2)))
    #        self.model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.2))

    # Conv. 5
    model.add(layers.Conv2D(96, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones'))

    # Max Pooling
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    #        self.model.add(layers.Dropout(0.2))

    # Conv. 6,7,8 & 9
    model.add(layers.Conv2D(96, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(96, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(96, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(96, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.2))

    # Conv. 10
    model.add(layers.Conv2D(144, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones'))

    # Max Pooling
    #model.add(layers.MaxPool2D(pool_size=(2, 2)))
    #        self.model.add(layers.Dropout(0.3))


    # Conv. 11
    model.add(layers.Conv2D(144, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.3))

    # Conv. 12
    model.add(layers.Conv2D(178, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.3))

    # Conv. 13


    model.add(layers.Conv2D(128, 3, strides=(1, 1), padding='valid', activation='relu', use_bias=True,
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    #        self.model.add(layers.Dropout(0.3))

    # Global Ma Pooling
    #model.add(layers.MaxPool2D(pool_size=(2, 2)))
    # Classifier
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    #model.add(layers.Dropout(0.5))

    if qlearning:
        model.add(layers.Dense(action_shape, activation='linear'))

        optmz = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=optmz,
                      loss='mean_squared_error',
                      # loss='binary_crossentropy',
                      metrics=['accuracy'])
    else:
        model.add(layers.Dense(action_shape, activation='softmax'))

        optmz = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=optmz,
                      # loss='categorical_crossentropy',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    model.summary()
    return model
