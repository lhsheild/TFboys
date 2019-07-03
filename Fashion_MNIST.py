from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


def plot_image(i, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    import math
    import matplotlib.pyplot as plt

    import tqdm
    import tqdm.auto

    tqdm.tqdm = tqdm.auto.tqdm
    print(tf.__version__)
    tf.enable_eager_execution()

    dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # print(dataset, metadata)

    num_train_examples = metadata.splits['train'].num_examples
    num_test_examples = metadata.splits['test'].num_examples
    print("Number of training examples: {}".format(num_train_examples))
    print("Number of test examples:     {}".format(num_test_examples))
    train_dataset = train_dataset.map(normalize)
    test_dataset = test_dataset.map(normalize)

    plt.figure(figsize=(10, 10))
    i = 0
    for (image, label) in test_dataset.take(25):
        image = image.numpy().reshape((28, 28))
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(class_names[label])
        i += 1
    plt.show()

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    BATCH_SIZE = 32
    train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples / BATCH_SIZE))

    test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples / 32))
    print('测试精确度：', test_accuracy)

    for test_images, test_labels in test_dataset.take(1):
        test_images = test_images.numpy()
        test_labels = test_labels.numpy()
        predictions = model.predict(test_images)

    predictions.shape
    print(predictions[0])
    print(np.argmax(predictions[0]))

    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test_labels)
    plt.show()