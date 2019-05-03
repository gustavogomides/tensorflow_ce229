# Import MINST data
import tensorflow as tf
from util import plot_data, get_mnist, print_predict_image
import numpy as np
from time import time

start_time = time()

tf.reset_default_graph()

x_train, y_train, y_train_classes, x_test, y_test = get_mnist()

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10
# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.2  # Dropout, probability to keep units
# tf Graph input
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
# dropout (keep probability)
keep_prob = tf.placeholder(tf.float32)
# Create model


def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add
                      (tf.nn.conv2d(img, w,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME'), b))


def max_pool(img, k):
    return tf.nn.max_pool(img,
                          ksize=[1, k, k, 1],
                          strides=[1, k, k, 1],
                          padding='SAME')


# Store layers weight & bias
# 5x5 conv, 1 input, 32 outputs
wc1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))
bc1 = tf.Variable(tf.random_normal([32]))
# 5x5 conv, 32 inputs, 64 outputs
wc2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
bc2 = tf.Variable(tf.random_normal([64]))
# fully connected, 7*7*64 inputs, 1024 outputs
wd1 = tf.Variable(tf.random_normal([7*7*64, 1024]))
# 1024 inputs, 10 outputs (class prediction)
wout = tf.Variable(tf.random_normal([1024, n_classes]))
bd1 = tf.Variable(tf.random_normal([1024]))

bout = tf.Variable(tf.random_normal([n_classes]))
# Construct model
# _X = tf.reshape(x, shape=[-1, 28, 28, 1])
# Convolution Layer
conv1 = conv2d(x, wc1, bc1)
# Max Pooling (down-sampling)
conv1 = max_pool(conv1, k=2)
# Convolution Layer
conv2 = conv2d(conv1, wc2, bc2)
# Max Pooling (down-sampling)
conv2 = max_pool(conv2, k=2)
# Fully connected layer
# Reshape conv2 output to fit dense layer input
dense1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
# Relu activation
dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, wd1), bd1))
# Apply Dropout
dense1 = tf.nn.dropout(dense1, keep_prob)
# Output, class prediction
pred = tf.add(tf.matmul(dense1, wout), bout)
# Define loss and optimizer
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer =\
    tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initializing the variables
init = tf.initialize_all_variables()
# Launch the graph


def next_batch(images, labels, batch_size):
    """Return the next `batch_size` examples from this data set."""
    _index_in_epoch = 0
    start = _index_in_epoch
    _index_in_epoch += batch_size
    num_examples = images.shape[0]
    _epochs_completed = 0
    if _index_in_epoch > num_examples:
        # Finished epoch
        _epochs_completed += 1
        # Shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        images = images[perm]
        labels = labels[perm]
        # Start next epoch
        start = 0
        _index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = _index_in_epoch
    return images[start:end], labels[start:end]


with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = next_batch(x_train, y_train, batch_size)

        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs,
                                       y: batch_ys,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs,
                                                y: batch_ys,
                                                keep_prob: 1.})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs,
                                             y: batch_ys,
                                             keep_prob: 1.})
            print("Iter " + str(step*batch_size) +
                  ", Minibatch Loss= " +
                  "{:.6f}".format(loss) +
                  ", Training Accuracy= " +
                  "{:.5f}".format(acc))
        step += 1

    print("Optimization Finished!")
    print("Testing Accuracy:",
          sess.run(accuracy,
                   feed_dict={x: x_test,
                              y: y_test,
                              keep_prob: 1.}))

    end_time = time()

    print("Processing time: {} s".format(end_time - start_time))
