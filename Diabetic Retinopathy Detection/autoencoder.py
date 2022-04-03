from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# start with defining autoencoder class
class Autoencoder:
    """
    Autoencoder class with mirrored encoder and decoder parts
    """
    # constructor function
    def __init__(self,
                 input_shape,
                 conv_filters,  # number of filters for each layer
                 conv_kernels,  # number of kernels
                 conv_strides,
                 latent_space_dim  # size of latent dimension, bottleneck
                 ):
            # Assign all of arguments to instance variables
            # Initially we will be using MNIST dataset
            # [H, W, number of channels] -> number of channels = 1 for grayscale
            self.input_shape        = input_shape
            self.conv_filters       = conv_filters
            self.conv_kernels       = conv_kernels
            self.conv_strides        = conv_strides
            self.latent_space_dim   = latent_space_dim

            self.encoder = None
            self.decoder = None
            self.model   = None

            self._num_conv_layers   = len(conv_filters)
            self._shape_before_bottleneck= None

            self._build()

    # let us define a method which will build all the part of autoencoder (methods)
    # encoder
    # decoder
    # autoencoder

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    # Encoder method
    # 1. Input
    # 2. conv layers
    # 3. bottle neck layer
    # 4. a tensorflow model which will tie the above components
    # we will define functions for all the above 4 components
    def _build_encoder(self):
        encoder_input   = self._add_encoder_input()
        conv_layers     = self._add_conv_layers(encoder_input)
        bottleneck      = self._add_bottleneck(conv_layers)

        # the following line fpr model_input - will be used in the AE build section
        self._model_input = encoder_input

        self.encoder = Model(encoder_input,
                             bottleneck,
                             name='encoder')

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name= 'encoder_input')

    def _add_conv_layers(self, encoder_input):
        # create all the conv layers
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)

        return x

    def _add_conv_layer(self, layer_index, x):
        layer_numer = layer_index + 1

        conv_layer = Conv2D(filters     = self.conv_filters[layer_index],
                            kernel_size = self.conv_kernels[layer_index],
                            strides     = self.conv_strides[layer_index],
                            padding = 'same',
                            name = f'encoder_conv_layer{layer_numer}'
        )

        x = conv_layer(x)
        x = ReLU(name=f'encoder_relu_{layer_numer}')(x)
        x = BatchNormalization(name= f'encoder_bn_{layer_numer}')(x)

        return x

    def _add_bottleneck(self, x):

        self._shape_before_bottleneck = K.int_shape(x)[1:]      #[6, 4, 4,1]

        x = Flatten()(x)
        x = Dense(self.latent_space_dim, name='encoder_output')(x)

        return x

    ################################## DECODER START ################################################
    # Input
    # dense_layer
    # reshape
    # build de-convulation layer
    # output
    def _build_decoder(self):
        decoder_input           = self._add_decoder_input()
        dense_layer             = self._add_dense_layer(decoder_input)
        reshape_layer           = self._add_reshape_layer(dense_layer)
        conv_transpose_layers   = self._add_conv_transpose_layers(reshape_layer)
        decoder_output          = self._add_decoder_output(conv_transpose_layers)

        self.decoder = Model(decoder_input, decoder_output, name='decoder')

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name ='decoder_input')

    def _add_dense_layer(self, decoder_input):
        # we need to know the number of neurons for this layer
        num_neurons = np.prod(self._shape_before_bottleneck)       #[4,4,3]
        dense_layer = Dense(num_neurons, name='decoder_dense')(decoder_input)

        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        # [0, 1,2,3] -> [3,2,1,0]
        # [   1,2,3] -> [3,2,1]
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x =self._add_conv_transpose_layer(layer_index, x)

        return x

    # import Con2DTranspose layer
    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index

        conv_transpose_layer = Conv2DTranspose(
                                    filters     =self.conv_filters[layer_index],
                                    kernel_size =self.conv_kernels[layer_index],
                                    strides     =self.conv_strides[layer_index],
                                    padding     ='same',
                                    name        =f'decoder_conv_transpose_layer{layer_num}')

        x = conv_transpose_layer(x)
        x = ReLU(name = f'decoder_relu_{layer_num}')(x)
        x = BatchNormalization(name=f'decoder_bn_{layer_num}')(x)

        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
                                                filters=1,
                                                kernel_size =self.conv_kernels[0],
                                                strides     =self.conv_strides[0],
                                                padding     ='same',
                                                name        =f'decoder_conv_transpose_layer{self._num_conv_layers}')

        x = conv_transpose_layer(x)

        output_layer = Activation('sigmoid', name='sigmoid_layer')(x)

        return output_layer

    ################################## DECODER END ##########################################


    #################################### AE part ############################################
    def _build_autoencoder(self):

        # qs: where should we define the _model_input
        model_input     = self._model_input
        model_output    = self.decoder(self.encoder(model_input))
        self.model      = Model(model_input, model_output, name='autoencoder')

    #################################### AE End #############################################

    #################################### Compile ############################################
    # SGD... Adam
    # loss ... mean squared error
    def compile(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss  = MeanSquaredError()

        # compile
        self.model.compile(optimizer=optimizer, loss= mse_loss)

    ################################## Compile END ##########################################

    ################################## Train part ###########################################
    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       verbose=1)

    #########################################################################################

    ################################# Saving the model ######################################
    

    ################################# Saving END ############################################

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

if __name__ == "__main__":
    autoencoder = Autoencoder(input_shape = (28, 28, 1),
                             conv_filters = (32, 64, 64, 64),
                             conv_kernels =  (3,3,3,3),
                             conv_strides = (1,2,2,1),
                             latent_space_dim  = 2)

    autoencoder.summary()
