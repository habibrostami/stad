from PIL import Image,ImageFile

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

from keras import regularizers


from keras import layers
from keras import models
import tensorflow as tf


def micronet(state_shape, action_shape,qlearning = False):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    NODE_NUM = 64
    DROP_OUT_RATE = 0.5
    #backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    init = tf.keras.initializers.he_uniform()

    #channel_axis = 'channels_first'

    g_kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)# regularizers.l2(1e-3) #None #regularizers.l1_l2(l1=1e-5, l2=1e-4),
    g_bias_regularizer=None #regularizers.l2(1e-4),
    g_activity_regularizer=regularizers.l2(1e-3)

    img_input = layers.Input(shape=state_shape)

    x = layers.Conv2D(16, (3, 3),
                      strides=(2, 2),
                      use_bias=False,
                      name='block1_conv1',
                      kernel_regularizer = g_kernel_regularizer,
                      bias_regularizer = g_bias_regularizer,
                      activity_regularizer = g_activity_regularizer)(img_input)

    x = layers.BatchNormalization( name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)
    #x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    #x = layers.BatchNormalization( name='block1_conv2_bn')(x)
    #x = layers.Activation('relu', name='block1_conv2_act')(x)

    residual = layers.Conv2D(128, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False,
                             kernel_regularizer=None, #g_kernel_regularizer,
                             #bias_regularizer=g_bias_regularizer,
                             activity_regularizer=None #g_activity_regularizer,
                             )(x)
    residual = layers.BatchNormalization()(residual)

    #x = layers.SeparableConv2D(128, (3, 3),
    #                           padding='same',
   #                            use_bias=False,
     #                          name='block2_sepconv1',
    #                           kernel_regularizer=g_kernel_regularizer,
   #                            #bias_regularizer=g_bias_regularizer,
    #                           activity_regularizer=g_activity_regularizer,
    #                           )(x)
   # x = layers.BatchNormalization( name='block2_sepconv1_bn')(x)
  #  x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv2')(x)
    x = layers.BatchNormalization( name='block2_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False,
                             kernel_regularizer=g_kernel_regularizer,
                             #bias_regularizer=g_bias_regularizer,
                             activity_regularizer=g_activity_regularizer,
                             )(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    #x = layers.SeparableConv2D(256, (3, 3),
     #                          padding='same',
   #                            use_bias=False,
   #                            name='block3_sepconv1')(x)
   # x = layers.BatchNormalization( name='block3_sepconv1_bn')(x)
   # x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),#256
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv2')(x)
    x = layers.BatchNormalization( name='block3_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(NODE_NUM, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False,
                             kernel_regularizer=g_kernel_regularizer,
                             #bias_regularizer=g_bias_regularizer,
                             activity_regularizer=g_activity_regularizer,
                             )(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    #x = layers.SeparableConv2D(728, (3, 3),
    #                           padding='same',
    #                           use_bias=False,
    #                           name='block4_sepconv1',
    #                           kernel_regularizer = g_kernel_regularizer,
    #                           #bias_regularizer = g_bias_regularizer,
    #                           activity_regularizer = g_activity_regularizer,
    #                           )(x)
    #x = layers.BatchNormalization( name='block4_sepconv1_bn')(x)
    #x = layers.Activation('relu', name='block4_sepconv2_act')(x)

    #x = layers.Dropout(DROP_OUT_RATE)(x)

    x = layers.SeparableConv2D(NODE_NUM, (3, 3),#728
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv2')(x)
    x = layers.BatchNormalization( name='block4_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block4_pool')(x)
    x = layers.add([x, residual])
    # default is 8
    for i in range(1):
        residual = x
        prefix = 'block' + str(i + 5)

        x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
        #x = layers.SeparableConv2D(728, (3, 3),
        #                           padding='same',
        #                           use_bias=False,
        #                           name=prefix + '_sepconv1',
        #                           kernel_regularizer=g_kernel_regularizer,
        #                           #bias_regularizer=g_bias_regularizer,
        #                           activity_regularizer=g_activity_regularizer,
        #                           )(x)
        #x = layers.BatchNormalization(
        #                              name=prefix + '_sepconv1_bn')(x)
        #x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
        #x = layers.SeparableConv2D(728, (3, 3),
        #                           padding='same',
        #                           use_bias=False,
        #                           name=prefix + '_sepconv2',
        #                           kernel_regularizer=g_kernel_regularizer,
        #                           #bias_regularizer=g_bias_regularizer,
        #                           activity_regularizer=g_activity_regularizer,
        #                           )(x)
        #x = layers.BatchNormalization(
        #                              name=prefix + '_sepconv2_bn')(x)
        #x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)

        #x = layers.Dropout(0.5)(x)

        x = layers.SeparableConv2D(NODE_NUM, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv3',
                                   kernel_regularizer=g_kernel_regularizer,
                                   #bias_regularizer=g_bias_regularizer,
                                   activity_regularizer=g_activity_regularizer,
                                   )(x)
        x = layers.BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    residual = layers.Conv2D(128, (1, 1), strides=(2, 2),#1025
                             padding='same', use_bias=False,
                             kernel_regularizer=g_kernel_regularizer,
                             #bias_regularizer=g_bias_regularizer,
                             activity_regularizer=g_activity_regularizer,
                             )(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block13_sepconv1_act')(x)
    # for best accu remove the following comment
    #x = layers.SeparableConv2D(728, (3, 3),
    #                           padding='same',
    #                           use_bias=False,
    #                           name='block13_sepconv1',
    #                           kernel_regularizer=g_kernel_regularizer,
    #                           #bias_regularizer=g_bias_regularizer,
    #                           activity_regularizer=g_activity_regularizer,
    #                           )(x)
    #x = layers.BatchNormalization( name='block13_sepconv1_bn')(x)
    #x = layers.Activation('relu', name='block13_sepconv2_act')(x)

    #x = layers.Dropout(DROP_OUT_RATE)(x)


    x = layers.SeparableConv2D(128, (3, 3),#1024
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv2',
                               kernel_regularizer=g_kernel_regularizer,
                               #bias_regularizer=g_bias_regularizer,
                               activity_regularizer=g_activity_regularizer,
                               )(x)
    x = layers.BatchNormalization( name='block13_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block13_pool')(x)
    x = layers.add([x, residual])

    #x = layers.SeparableConv2D(1536, (3, 3), #1536
    #                           padding='same',
    #                           use_bias=False,
    #                           name='block14_sepconv1',
    #                           kernel_regularizer=g_kernel_regularizer,
    #                           #bias_regularizer=g_bias_regularizer,
    #                           activity_regularizer=g_activity_regularizer,
    #                           )(x)
    #x = layers.BatchNormalization( name='block14_sepconv1_bn')(x)
    #x = layers.Activation('relu', name='block14_sepconv1_act')(x)

    #x = layers.Dropout(DROP_OUT_RATE)(x)

    #x = layers.SeparableConv2D(2048, (3, 3),
    #                           padding='same',
    #                           use_bias=False,
    #                           name='block14_sepconv2',
    #                           kernel_regularizer=g_kernel_regularizer,
    #                           #bias_regularizer=g_bias_regularizer,
    #                           activity_regularizer=g_activity_regularizer,
    #                           )(x)
    #x = layers.BatchNormalization( name='block14_sepconv2_bn')(x)
    #x = layers.Activation('relu', name='block14_sepconv2_act')(x)

    #x = layers.Dropout(DROP_OUT_RATE)(x)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    #x = layers.Flatten()(x)


    if qlearning:
        x = layers.Dense(action_shape, activation='linear', name='predictions', kernel_initializer=init,
                      kernel_regularizer = g_kernel_regularizer,
                      bias_regularizer = g_bias_regularizer,
                      activity_regularizer = g_activity_regularizer,
                      )(x)
    else:
        x = layers.Dense(action_shape, activation='softmax', name='predictions', kernel_initializer=init,
                         kernel_regularizer=g_kernel_regularizer,
                         bias_regularizer=g_bias_regularizer,
                         activity_regularizer=g_activity_regularizer,
                         )(x)


    inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='xception')

    return model


