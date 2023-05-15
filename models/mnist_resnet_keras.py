import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, utils

def identity_block(X, filters, kernel_size):
    F1, F2 = filters
    
    X_shortcut = X
    
    X = layers.Conv2D(filters=F1, kernel_size=kernel_size, strides=(1, 1), padding='same')(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(filters=F2, kernel_size=kernel_size, strides=(1, 1), padding='same')(X)
    X = layers.BatchNormalization(axis=3)(X)

    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)

    return X

def convolutional_block(X, filters, kernel_size, strides):
    F1, F2 = filters

    X_shortcut = X
    
    X = layers.Conv2D(filters=F1, kernel_size=kernel_size, strides=strides, padding='same')(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(filters=F2, kernel_size=kernel_size, strides=(1, 1), padding='same')(X)
    X = layers.BatchNormalization(axis=3)(X)

    X_shortcut = layers.Conv2D(filters=F2, kernel_size=(1, 1), strides=strides, padding='valid')(X_shortcut)
    X_shortcut = layers.BatchNormalization(axis=3)(X_shortcut)

    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)

    return X

def resnet18_mnist(input_shape, num_classes):
    X_input = layers.Input(input_shape)

    X = layers.ZeroPadding2D(padding=(2, 2))(X_input)
    X = layers.Conv2D(64, (7, 7), strides=(1, 1))(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    X = convolutional_block(X, filters=[64, 64], kernel_size=(3, 3), strides=(2, 2))
    X = identity_block(X, filters=[64, 64], kernel_size=(3, 3))

    X = convolutional_block(X, filters=[128, 128], kernel_size=(3, 3), strides=(2, 2))
    X = identity_block(X, filters=[128, 128], kernel_size=(3, 3))

    X = convolutional_block(X, filters=[256, 256], kernel_size=(3, 3), strides=(2, 2))
    X = identity_block(X, filters=[256, 256], kernel_size=(3, 3))

    X = convolutional_block(X, filters=[512, 512], kernel_size=(3, 3), strides=(2, 2))
    X = identity_block(X, filters=[512, 512], kernel_size=(3, 3))

    X = layers.GlobalAveragePooling2D()(X)
    X = layers.Dense(num_classes, activation='softmax')(X)

    model = models.Model(inputs=X_input, outputs=X, name='ResNet18_MNIST')

    return model
