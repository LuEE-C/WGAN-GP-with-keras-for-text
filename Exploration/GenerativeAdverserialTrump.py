from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, BatchNormalization, Dropout, GRU, Bidirectional, Reshape, Activation
from keras.layers.noise import GaussianNoise
from keras.models import Model
from keras.layers.merge import _Merge
from keras.layers import Convolution1D, AveragePooling1D, ZeroPadding1D, UpSampling1D
from keras.optimizers import Adam
from random import randint
from keras import backend as K
from keras import layers
import numpy as np
from functools import partial
from Exploration.data_util import text_from_seed, convert_text_to_nptensor
import pickle

# Data taken from https://github.com/bpb27/trump_tweet_data_archive

BATCH_SIZE = 256
TRAINING_RATIO = 5
GRADIENT_PENALTY_WEIGHT = 1

def res_cnn(shape1, shape2):
    input_tensor = Input(shape=(shape1, shape2))
    x = Convolution1D(filters=shape2, kernel_size=3, padding='same', activation='elu')(input_tensor)
    x = BatchNormalization()(x)
    x = Convolution1D(filters=shape2, kernel_size=3, padding='same', activation='elu')(x)
    x = BatchNormalization()(x)
    x = layers.add([input_tensor, x])
    output = Activation('elu')(x)
    #output = BatchNormalization()(x)
    res_1d = Model(inputs=[input_tensor], outputs=[output])
    return res_1d


# Loss functions taken from https://github.com/farizrahman4u/keras-contrib/blob/master/examples/improved_wgan.py
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(K.sum(y_pred), averaged_samples)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return gradient_penalty


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def generator_mod(x_shape_1, x_shape_2, conv_shape):
    model = Sequential()
    model.add(Dense(x_shape_1, activation='elu', input_shape=(100,)))
    model.add(Reshape((x_shape_1, 1)))
    model.add(Convolution1D(conv_shape, 7, padding='same'))
    model.add(res_cnn(x_shape_1, conv_shape))
    model.add(res_cnn(x_shape_1, conv_shape))
    model.add(res_cnn(x_shape_1, conv_shape))
    model.add(res_cnn(x_shape_1, conv_shape))
    model.add(res_cnn(x_shape_1, conv_shape))
    # # This is for numeric stability, elu + softmax are not friends
    # model.add(Activation('tanh'))
    model.add(Dense(x_shape_2, activation='softmax'))
    return model


def discriminator_mod(x_shape_1, x_shape_2, conv_shape):
    model = Sequential()
    model.add(Dense(x_shape_1, activation='elu', input_shape=(x_shape_1, x_shape_2)))
    model.add(Convolution1D(conv_shape, 7, padding='same'))
    model.add(res_cnn(x_shape_1, conv_shape))
    model.add(res_cnn(x_shape_1, conv_shape))
    model.add(res_cnn(x_shape_1, conv_shape))
    model.add(res_cnn(x_shape_1, conv_shape))
    model.add(res_cnn(x_shape_1, conv_shape))
    model.add(Flatten())
    model.add(Dense(256, activation='elu'))
    model.add(Dense(256, activation='elu'))
    model.add(Dense(1))
    return model


def train(from_save_point=False, suffix='rnn'):
    X_train = np.load('../text_' + suffix + '.npy')
    print(X_train.shape)
    with open('../ind_to_word_' + suffix + '.pickle', 'rb') as pkl:
        ind_to_word = pickle.load(pkl)

    generator = generator_mod(X_train.shape[1], X_train.shape[2], 256)
    discriminator = discriminator_mod(X_train.shape[1], X_train.shape[2], 256)
    generator.summary()
    discriminator.summary()

    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False

    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False
    generator_input = Input(shape=(100,))
    generator_layers = generator(generator_input)
    discriminator_layers_for_generator = discriminator(generator_layers)
    generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
    generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)

    for layer in discriminator.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    discriminator.trainable = True
    generator.trainable = False

    real_samples = Input(shape=X_train.shape[1:])
    generator_input_for_discriminator = Input(shape=(100,))
    generated_samples_for_discriminator = generator(generator_input_for_discriminator)
    discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
    discriminator_output_from_real_samples = discriminator(real_samples)

    averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
    averaged_samples_out = discriminator(averaged_samples)
    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
    partial_gp_loss.__name__ = 'gradient_penalty'

    discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                                outputs=[discriminator_output_from_real_samples,
                                         discriminator_output_from_generator,
                                         averaged_samples_out])
    discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                                loss=[wasserstein_loss,
                                      wasserstein_loss,
                                      partial_gp_loss])
    positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

    if from_save_point:
        generator.load_weights('generator_' + suffix)
        discriminator.load_weights('discriminator_' + suffix)

    for epoch in range(100000):
        np.random.shuffle(X_train)
        discriminator_loss = []
        generator_loss = []
        minibatches_size = BATCH_SIZE * TRAINING_RATIO
        X_seed = np.random.normal(size=(X_train.shape[0], 100))

        for i in range(int(X_train.shape[0] // (BATCH_SIZE * TRAINING_RATIO))):

            discriminator_minibatches = X_train[i * minibatches_size:(i + 1) * minibatches_size]
            for j in range(TRAINING_RATIO):
                image_batch = discriminator_minibatches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
                start_prediction = randint(0, X_seed.shape[0] - 1 - BATCH_SIZE)
                noise = X_seed[start_prediction:start_prediction + BATCH_SIZE]
                discriminator_loss.append(discriminator_model.train_on_batch([image_batch, noise],
                                                                             [positive_y, negative_y, dummy_y]))
            start_prediction = randint(0, X_seed.shape[0] - 1 - BATCH_SIZE)
            generator_loss.append(generator_model.train_on_batch(X_seed[start_prediction:start_prediction + BATCH_SIZE], positive_y))
        if epoch % 1 == 0:
            print("Epoch: ", epoch)
            generator.save_weights('generator_' + suffix, True)
            discriminator.save_weights('discriminator_' + suffix, True)
            print(generator_loss[-1], discriminator_loss[-1], "% done : ", i // (BATCH_SIZE * TRAINING_RATIO) * 100.0)
            text_from_seed(model=generator, name="../Data/sample_" + str(epoch) + '_' + suffix + ".txt",
                           ind_to_char=ind_to_word, char=False)


if __name__ == "__main__":
    # convert_text_to_nptensor(cutoff=50, min_frequency_words=100000, max_lines=20000000)
    train(from_save_point=False, suffix='Google')