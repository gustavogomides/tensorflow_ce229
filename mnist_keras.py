import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from util import plot_data, get_mnist
from time import time

start_time = time()

x_train, y_train, y_train_classes, x_test, y_test = get_mnist()

# plot_data(5, x_train, y_train_classes)


def build_cnn():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',
                     input_shape=(28, 28, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=[2, 2], strides=2))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=[2, 2], strides=2))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])

    # model.summary()

    return model


model = build_cnn()

model.fit(x_train, y_train,
          batch_size=128,
          epochs=3,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# print_predict_image(5, x_test, model, 'keras')

end_time = time()

print("Processing time: {} s".format(end_time - start_time))
