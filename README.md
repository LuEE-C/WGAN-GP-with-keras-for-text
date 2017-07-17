# WGAN-GP-with-keras-for-text
My implementation of the 1d convolutional wgan described in this paper https://arxiv.org/abs/1705.10929.

Simply download the data from http://www.statmt.org/lm-benchmark/ and run the convert_text_to_nptensor function.

Keras and either Theano, Tensorflow or CNTK are required. I believe Theano is the only one that lets you calculate second order gradient, so using the loss function with any RNN will only work with a Theano backend.
