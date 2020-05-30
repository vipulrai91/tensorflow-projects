# A dense layer implements the following input transformations  w, b - model parameters
# activation is elementwise function (relu, last layer  will be softmax)

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np


class NaiveDense:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation

        w_shape = (input_size, output_size)
        # (2,3)

        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
        # tf.Tensor(
        # [[0.01800234 0.0264619  0.08320428]
        # [0.06735289 0.08069873 0.02565409]], shape=(2, 3), dtype=float32)

        self.W = tf.Variable(w_initial_value)
        # tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=
        # array([[0.02179157, 0.04815897, 0.02990031],
        #     [0.01661453, 0.08813614, 0.09748103]], dtype=float32)>

        b_shape = (output_size,)
        # (3,)

        b_initial_value = tf.zeros(b_shape)
        # tf.Tensor([0. 0. 0.], shape=(3,), dtype=float32)

        self.b = tf.Variable(b_initial_value)
        # <tf.Variable 'Variable:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>

    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b)

    @property
    def weights(self):
        return [self.W, self.b]

    # [<tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=
    #     array([[0.0920585 , 0.08967658, 0.01914843],
    #    [0.01162399, 0.04173064, 0.07235775]], dtype=float32)>, <tf.Variable 'Variable:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>]


class NaiveSequential:
    """Next we will create a Sequential class to chain these layers"""

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights


model = NaiveSequential(
    [
        NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
        NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax),
    ],
)

assert len(model.weights) == 4


class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size

    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels


LEARNING_RATE = 1e-3


def update_weights(gradients, weights):
    "We can also use optimizer from keras"
    for g, w in zip(gradients, model.weights):
        w.assign_sub(w * LEARNING_RATE)


def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
            labels_batch, predictions
        )
        average_loss = tf.reduce_mean(per_sample_losses)
        gradients = tape.gradient(average_loss, model.weights)
        update_weights(gradients, model.weights)
        return average_loss


def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch : {epoch_counter}")
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(len(images) // batch_size):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"loss at batch {batch_counter} {loss}")


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

fit(model, train_images, train_labels, epochs=3, batch_size=128)


predictions = model(test_images)
predictions = predictions.numpy()
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print("accuracy: %.2f" % np.mean(matches))
