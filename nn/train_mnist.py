import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


from tensorflow.keras.datasets import mnist

# Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

print(f"Training data shape: {train_images.shape}")
print(f"Test data shape: {test_images.shape}")

#WIP
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])

model.load_weights('.weights.h5')

model.compile(
    optimizer = 'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# model.fit(
#   train_images, # training data
#   to_categorical(train_labels), # training targets
#   epochs=5,
#   batch_size=32,
# )

#model.save_weights('.weights.h5')

model.evaluate(
  test_images,
  to_categorical(test_labels)
)
predictions = model.predict(test_images[:20])

print(np.argmax(predictions, axis=1))

print(test_labels[:20])
