from util import plot_data, get_mnist, print_predict_image
import tensorflow as tf
import keras
from time import time

tf.reset_default_graph()

start_time = time()

x_train, y_train, y_train_classes, x_test, y_test = get_mnist()

# plot_data(5, x_train, y_train_classes)


def build_cnn(features, labels, mode):
    input_layer = features['X']

    with tf.name_scope("conv1"):
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], activation=tf.nn.relu,
                                 padding='same')

    with tf.name_scope("pool1"):
        pooling1 = tf.layers.max_pooling2d(
            inputs=conv1, pool_size=[2, 2], strides=2)

    with tf.name_scope("conv2"):
        conv2 = tf.layers.conv2d(inputs=pooling1, filters=64, kernel_size=[5, 5], activation=tf.nn.relu,
                                 padding='same')

    with tf.name_scope("pool2"):
        pooling2 = tf.layers.max_pooling2d(
            inputs=conv2, pool_size=[2, 2], strides=2)

    with tf.name_scope("flatten"):
        flattening = tf.reshape(pooling2, [-1, 7 * 7 * 64])

    with tf.name_scope("dense"):
        dense = tf.layers.dense(
            inputs=flattening, units=1024, activation=tf.nn.relu)

    with tf.name_scope("dropout"):
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    with tf.name_scope("output"):
        output_layer = tf.layers.dense(inputs=dropout, units=10)

        predicts = tf.argmax(output_layer, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predicts)

    error = tf.losses.softmax_cross_entropy(
        onehot_labels=labels, logits=output_layer)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(
            error, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=error, train_op=train)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'acc': tf.metrics.accuracy(
                tf.argmax(input=output_layer, axis=1),
                tf.argmax(input=labels, axis=1))
        }

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predicts,
                                          loss=error, eval_metric_ops=eval_metric_ops)


classifier = tf.estimator.Estimator(
    model_fn=build_cnn, model_dir='./mnist_tf_estimator')

input_fn = tf.estimator.inputs.numpy_input_fn(x={'X': x_train}, y=y_train, batch_size=128,
                                              num_epochs=None, shuffle=True)

classifier.train(input_fn=input_fn, steps=200)

test_fn = tf.estimator.inputs.numpy_input_fn(x={'X': x_test}, y=y_test, num_epochs=1,
                                             shuffle=False)
results = classifier.evaluate(input_fn=test_fn)
print(results)

tf.logging.set_verbosity(tf.logging.ERROR)
# print_predict_image(5, x_test, classifier, 'tf')

end_time = time()

print("Processing time: {} s".format(end_time - start_time))
