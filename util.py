from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras


def plot(qty, images, labels):
    for index, (img, label) in enumerate(zip(images, labels)):
        plt.subplot(2, qty, index + 1)
        plt.axis('off')
        plt.imshow(np.reshape(img, (28, 28)),
                   cmap='gray', interpolation='nearest')
        plt.title('%i' % label)
    plt.show()


def plot_data(qty, x_train, y_train):
    random_indices = np.random.randint(0, x_train.shape[0], qty)

    images = x_train[random_indices]
    labels = y_train[random_indices]

    plot(qty, images, labels)


def get_mnist():
    # read dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path='./mnist.npz')

    # input image dimensions
    img_rows, img_cols = 28, 28
    num_classes = 10

    # reshape data
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    # convert type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')

    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train_classes = y_train
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print('Train labels:', y_train.shape[0])
    print('Test labels:', y_test.shape[0])

    return x_train, y_train, y_train_classes, x_test, y_test


def predict_digit(x_test, classifier, lib):
    x_image_test = x_test[np.random.randint(
        0, x_test.shape[0])].reshape(1, 28, 28, -1)

    if lib == 'tf':
        predict_fn = tf.estimator.inputs.numpy_input_fn(
            x={'X': x_image_test}, shuffle=False)

        return list(classifier.predict(input_fn=predict_fn)), x_image_test

    return classifier.predict_classes(x_image_test), x_image_test


def print_predict_image(qty, x_test, classifier, lib):

    images = []
    labels = []
    for _ in range(qty):
        pred, x_image_test = predict_digit(x_test, classifier, lib)
        images.append(x_image_test)
        labels.append(pred[0])

    plot(qty, images, labels)
