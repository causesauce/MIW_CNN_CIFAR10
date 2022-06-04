# %% importing modules
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %% data preprocessing
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
if X_train.shape[0] + X_test.shape[0] != y_train.shape[0] + y_test.shape[0]:
    raise Exception("number of records is not consistent")

# train_set_len = int((X_train.shape[0] + X_test.shape[0]) * 0.7)

X = np.concatenate((X_train, X_test)) / 225.0
y = np.concatenate((y_train, y_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# %% show first image
plt.imshow(X_train[0])
plt.show()

# %% importing and setting model
from keras import models, layers

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()

# %% compiling and running model
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# %% model evaluation

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(test_acc)
# acc is 0.5885555744171143