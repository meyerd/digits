#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Dominik Meyer <meyerd@mytum.de>'

import os
import sys
import math
import mnist
import numpy as np
import lasagne
import theano
import theano.tensor as T
import progressbar
import time


def load_mnist_data_lasagne_example():
    datadir = os.path.join("..", "data", "mnist")
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, os.path.join(datadir, filename))

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(os.path.join(datadir, filename)):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(os.path.join(datadir, filename), 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(os.path.join(datadir, filename)):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(os.path.join(datadir, filename), 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_cnn(input_var=None):
    network = []
    input_layer = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                            input_var=input_var)
    network.append(input_layer)
    conv2d1 = lasagne.layers.Conv2DLayer(
        input_layer, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    network.append(conv2d1)
    maxpool1 = lasagne.layers.MaxPool2DLayer(conv2d1, pool_size=(2, 2))
    network.append(maxpool1)
    conv2d2 = lasagne.layers.Conv2DLayer(
        maxpool1, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify)
    network.append(conv2d2)
    maxpool2 = lasagne.layers.MaxPool2DLayer(conv2d2, pool_size=(2, 2))
    network.append(maxpool2)
    dense1 = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(maxpool2, p=0.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)
    network.append(dense1)
    dense2 = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(dense1, p=0.5),
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)
    network.append(dense2)

    return network, dense2


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def save_model_for_js(model, filename):
    import json
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    # NOTE: this only works for the above cnn model and the following convnetjs model:
    #  layer_defs.push({type: 'input', out_sx: 28, out_sy: 28, out_depth: 1});
    #  layer_defs.push({type: 'conv', filters: 32, sx: 5, stride: 1, activation: 'relu'});
    #  layer_defs.push({type: 'pool', sx: 2, stride: 2});
    #  layer_defs.push({type: 'conv', filters: 32, sx: 5, stride: 1, activation: 'relu'});
    #  layer_defs.push({type: 'pool', sx: 2, stride: 2});
    #  layer_defs.push({type: 'fc', num_neurons: 256, activation: 'relu'});
    #  layer_defs.push({type: 'softmax', num_classes: 10});

    params = lasagne.layers.get_all_param_values(model)
    outobj = list()
    outobj.append({
        'arraypos': 1,
        'filters': [params[0][i, :].ravel().tolist() for i in range(32)],
        'filtersize': 25,
        'biases': params[1].tolist()
    })
    outobj.append({
        'arraypos': 4,
        # 'filters': [params[2][i, :].reshape(32, -1).swapaxes(0, 1).ravel().tolist() for i in range(32)],
        'filters': [params[2][i, :].reshape(32, -1).ravel().tolist() for i in range(32)],
        'filtersize': 800,
        'biases': params[3].tolist()
    })
    outobj.append({
        'arraypos': 7,
        'filters': [params[4][:, i].ravel().tolist() for i in range(256)],
        'filtersize': 512,
        'biases': params[5].tolist()
    })
    outobj.append({
        'arraypos': 9,
        'filters': [params[6][:, i].ravel().tolist() for i in range(10)],
        'filtersize': 256,
        'biases': params[7].tolist()
    })
    with open(filename, 'w') as f:
        json.dump(outobj, f)


def main(model='cnn', num_epochs=500):
    print("loading data ...")
    # X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data_lasagne_example()

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    print("building model and compiling functions ...")
    if model == 'cnn':
        network, network_output = build_cnn(input_var=input_var)
    else:
        print("Unknown model type %r." % model)
        sys.exit(1)

    prediction = lasagne.layers.get_output(network_output)
    loss = lasagne.objectives.categorical_crossentropy(predictions=prediction, targets=target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network_output, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
    updates = lasagne.updates.adam(loss, params)

    test_prediction = lasagne.layers.get_output(network_output, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    print("training ...")
    batchsize = 500
    # bar = progressbar.ProgressBar(redirect_stdout=False)
    bar = progressbar.ProgressBar(redirect_stdout=True)
    for epoch in bar(range(num_epochs)):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batchsize, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batchsize, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))

    save_model_for_js(network_output, "convnetjsmodel.json")

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

if __name__ == '__main__':
    main(model='cnn', num_epochs=500)

