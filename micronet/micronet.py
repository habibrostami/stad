from PIL import Image,ImageFile

from keras import regularizers


from keras import layers
from keras import models
import tensorflow as tf

def micronet(state_shape, action_shape,qlearning = False, include_top = True):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    NODE_NUM = 512
    DROP_OUT_RATE = 0.5
    init = tf.keras.initializers.glorot_normal()#he_uniform()


    g_kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2)# regularizers.l2(1e-3) #None #regularizers.l1_l2(l1=1e-5, l2=1e-4),
    g_bias_regularizer=None #regularizers.l2(1e-4),
    g_activity_regularizer=regularizers.l2(1e-2)

    img_input = layers.Input(shape=state_shape)

    x = layers.Conv2D(32, (3, 3),
                      strides=(2, 2),
                      use_bias=False,kernel_initializer=init,
                      name='block1_conv1',
                      kernel_regularizer = g_kernel_regularizer,
                      bias_regularizer = g_bias_regularizer,
                      activity_regularizer = g_activity_regularizer)(img_input)

    x = layers.BatchNormalization( name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)

    residual = layers.Conv2D(32, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False,kernel_initializer=init,
                             kernel_regularizer=None, #g_kernel_regularizer,
                             #bias_regularizer=g_bias_regularizer,
                             activity_regularizer=None #g_activity_regularizer,
                             )(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(32, (3, 3),
                               padding='same',
                               use_bias=False,
                              name='block2_sepconv1',kernel_initializer=init,
                              kernel_regularizer=g_kernel_regularizer,
                              #bias_regularizer=g_bias_regularizer,
                              activity_regularizer=g_activity_regularizer,
                              )(x)
    x = layers.BatchNormalization( name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv2D(32, (3, 3),
                               padding='same',
                               use_bias=False,kernel_initializer=init,
                               name='block2_sepconv2')(x)
    x = layers.BatchNormalization( name='block2_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(64, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False,kernel_initializer=init,
                             kernel_regularizer=g_kernel_regularizer,
                             #bias_regularizer=g_bias_regularizer,
                             activity_regularizer=g_activity_regularizer,
                             )(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConv2D(64, (3, 3),
                               padding='same',
                               use_bias=False,kernel_initializer=init,
                               name='block3_sepconv1')(x)
    x = layers.BatchNormalization( name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv2D(64, (3, 3),#256
                               padding='same',
                               use_bias=False,kernel_initializer=init,
                               name='block3_sepconv2')(x)
    x = layers.BatchNormalization( name='block3_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(NODE_NUM, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False,kernel_initializer=init,
                             kernel_regularizer=g_kernel_regularizer,
                             #bias_regularizer=g_bias_regularizer,
                             activity_regularizer=g_activity_regularizer,
                             )(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)

    x = layers.SeparableConv2D(NODE_NUM, (3, 3),#728
                               padding='same',
                               use_bias=False,kernel_initializer=init,
                               name='block4_sepconv2')(x)
    x = layers.BatchNormalization( name='block4_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block4_pool')(x)
    x = layers.add([x, residual])
    # default is 8
    residual = x
    prefix = 'block_6_x1'

    x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)

    x = layers.SeparableConv2D(NODE_NUM, (3, 3),
                               padding='same',
                               use_bias=False,kernel_initializer=init,
                               name=prefix + '_sepconv3',
                               kernel_regularizer=g_kernel_regularizer,
                               #bias_regularizer=g_bias_regularizer,
                               activity_regularizer=g_activity_regularizer,
                               )(x)
    x = layers.BatchNormalization(name=prefix + '_sepconv3_bn')(x)

    x = layers.add([x, residual])

    residual = layers.Conv2D(64, (1, 1), strides=(2, 2),#1025
                             padding='same', use_bias=False,kernel_initializer=init,
                             kernel_regularizer=g_kernel_regularizer,
                             #bias_regularizer=g_bias_regularizer,
                             activity_regularizer=g_activity_regularizer,
                             )(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block13_sepconv1_act')(x)

    #x = layers.Dropout(DROP_OUT_RATE)(x)


    x = layers.SeparableConv2D(64, (3, 3),#1024
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv2',kernel_initializer=init,
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

    #x = layers.Dropout(DROP_OUT_RATE)(x)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    #x = layers.Flatten()(x)

    if include_top:
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
    model = models.Model(inputs, x, name='simple_xception')
    return model



