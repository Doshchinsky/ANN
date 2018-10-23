#!/usr/bin/python3

from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.np_utils import np_utils

(train_X, trane_Y, x, y) = mnist.load()
train_Y = np_utils.to_categorical(train_y)
model = Sequential()
model.add(Dense(units = 50,
                activation = 'sigmoid',
                kernel_initializer = 'random.uniform',
                bias_initializer = 'zeros',
                input_dim = 150))

model.add(Dense(units = 50,
                activation = 'softmax',
                kernel_initializer = 'random.uniform',
                bias_initializer = 'zeros'))

model.compile(loss = 'categorial_crossentropy'
              optimizer = 'RMSprop',
              metrics = ['accuracy'])

model.fit(train_X,
         train_Y,
         epochs = 200,
         batch_size = 2)

#model.predict(x)
score = model.evaluate(y, x)
