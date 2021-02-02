"""Testing with training_pipeline"""

from typing import Tuple

from keras import Sequential, optimizers
from keras.models import Model
from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.convolutional import MaxPooling3D, AveragePooling3D
from keras.layers import Dropout, Input
from keras.layers import Flatten, add
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization  # batch Normalization for managing internal covariant shift.
from keras.layers import Activation
from keras.utils import plot_model

from modules.cnn.training_models.training_pipeline_3d import pipeline_from_command_line

def Conv3d_BN(x, nb_filter, kernel_size, strides=1, padding='same', name=None):
    x = Conv3D(nb_filter, kernel_size, padding=padding, data_format='channels_last', strides=strides,
               activation='relu')(x)
    x = BatchNormalization()(x)
    return x


def identity_Block(inpt, nb_filter, kernel_size, strides=1, with_conv_shortcut=False):
    x = Conv3d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv3d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv3d_BN(inpt, nb_filter=nb_filter, strides=strides,
                             kernel_size=kernel_size)
        x = Dropout(0.2)(x)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

# def bottlneck_Block(inpt, nb_filter, strides=1, with_conv_shortcut=False):
#     k1, k2, k3 = nb_filter
#     x = Conv3d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
#     x = Conv3d_BN(x, nb_filter=k2, kernel_size=3, padding='same')
#     x = Conv3d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
#     if with_conv_shortcut:
#         shortcut = Conv3D(inpt, nb_filter=k3, data_format='channels_first', strides=strides, kernel_size=1)
#         x = add([x, shortcut])
#         return x
#     else:
#         x = add([x, inpt])
#         return x

def create_3D_resnet_model(input_shape: Tuple[int, int, int, int, int]):
    print(1111, input_shape)
    inpt = Input(shape=input_shape)
    print(2222, inpt)
#    x = ZeroPadding3D((1, 1, 1), data_format='channels_last')(inpt)
#    print(33333, x.shape)

    # conv1
#    x = Conv3d_BN(x, nb_filter=16, kernel_size=(6, 6, 6), strides=1, padding='valid')
    x = Conv3d_BN(inpt, nb_filter=16, kernel_size=(6, 6, 6), strides=1, padding='valid')

    x = MaxPooling3D(pool_size=(3, 3, 3), strides=2, data_format='channels_last')(x)

    # conv2_x
    x = identity_Block(x, nb_filter=32, kernel_size=(2, 2, 2), strides=1, with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=32, kernel_size=(2, 2, 2))
#     x = identity_Block(x, nb_filter=64, kernel_size=(3, 3, 3))
    # conv3_x
    x = identity_Block(x, nb_filter=64, kernel_size=(2, 2, 2), strides=1, with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=64, kernel_size=(2, 2, 2))
#     x = identity_Block(x, nb_filter=128, kernel_size=(3, 3, 3))
#     x = identity_Block(x, nb_filter=128, kernel_size=(3, 3, 3))

#     # conv4_x
#     x = identity_Block(x, nb_filter=256, kernel_size=(3, 3, 3), strides=2, with_conv_shortcut=True)
#     x = identity_Block(x, nb_filter=256, kernel_size=(3, 3, 3))
#     x = identity_Block(x, nb_filter=256, kernel_size=(3, 3, 3))

    x = AveragePooling3D(pool_size=(2, 2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(2, activation='softmax')(x)#2 is the number of classes

    model = Model(inputs=inpt, outputs=x)
    
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.adam(lr=1e-5),
        metrics=["accuracy"],
    )
   
    
    return model   

if __name__ == "__main__":

    pipeline_from_command_line(create_3D_resnet_model, rgb=False)
