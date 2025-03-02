from tensorflow.keras.datasets import mnist
from conv import Conv3x3
from maxpool import maxpool2
from softmax import softmax
import numpy as np

# The mnist package handles the MNIST dataset for us!
# Learn more at https://github.com/datapythonista/mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
conv = Conv3x3(8)
pool = maxpool2()
softmax = softmax(13 * 13 * 8, 10)

def forward(image, label):

  out = conv.forward((image / 255) - 0.5)
  out = pool.forward(out)
  out = softmax.forward(out)

  # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0

  return out, loss, acc

print('MNIST CNN initialized!')

loss = 0
num_correct = 0
for i, (im, label) in enumerate(zip(test_images, test_labels)):
  # Do a forward pass.
  _, l, acc = forward(im, label)
  loss += l
  num_correct += acc

  # Print stats every 100 steps.
  if i % 100 == 99:
    print(
      '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
      (i + 1, loss / 100, num_correct)
    )
    loss = 0
    num_correct = 0