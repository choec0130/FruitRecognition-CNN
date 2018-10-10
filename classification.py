import cv2
import os
import numpy as np
import h5py as h5
import random

CLASSES = ["Apple Braeburn", "Apple Golden", "Apple Granny Smith", "Apple Red", "Avocado", "Apricot", "Banana", "Cactus",
            "Cantaloupe", "Carambula", "Cherry", "Clementine", "Cocos", "Grapes", "Dates", "Garandilla", "Grapefruit".
            "Guava", "Huckleberry", "Kiwi", "Kaki", "Kumquats", "Lemon", "Lime", "Lychee", "Mandarin", "Mango", "Orange",
            "Maracuja", "Melon", "Mulberry", "Nectarine", "Orange", "Papaya", "Passion Fruit", "Peach", "Pear", "Pepino",
            "Physalis", "Physalis with Husk", "Pineapple", "Pitahaya", "Plum", "Pomegranate", "Quince", "Rambutan", "Raspberry",
            "Salak", "Strawberry", "Tamarillo", "Tangelo", "Walnut"]
TRAINING_DIR = "fruits-360/Training/"
TEST_DIR = "fruits-360/Test/"



def image_to_array(image):
    #Convert an image path to an array that can be used for classification
    return cv2.imread(image)


def get_label(label, one_hot=False):
    if not one_hot:
        return CLASSES.index(label)
    vector = np.zeros(len(CLASSES))
    vector[(CLASSES.index(label))] = 1
    return vector


def get_training_data():
    try:
        training_file = h5.File(TRAINING_DIR + 'fruit_data.hdf5', 'r')
        return training_file["Training"]["images"], training_file["Training"]["labels"]
    except:
        print("Constructing training data from scratch")
        paths_and_labels = []
        for subdir, dirs, files in os.walk(TRAINING_DIR):
            for file in files:
                label = os.path.basename(subdir)
                if label in CLASSES:
                    path = os.path.join(subdir, file)
                    paths_and_labels.append((path, get_label(label)))

        random.shuffle(paths_and_labels)
        split_index = len(paths_and_labels) // 5

        training_half = paths_and_labels[split_index:]
        validation_half = paths_and_labels[:split_index]

        training_file = h5.File(TRAINING_DIR + 'fruit_data.hdf5', 'w')

        training_images, training_labels = format_data(training_half, training_file, "Training")
        validation_images, validation_labels = format_data(validation_half, training_file, "Validation")

        return training_images, training_labels


def get_test_data(batch_size):
    data = []
    for subdir, dirs, files in os.walk(TRAINING_DIR):
        for file in files:
            label = os.path.basename(subdir)
            if label in CLASSES:
                path = os.path.join(subdir, file)
                data.append((image_to_array(path), get_label(label)))

    test_images, test_labels = format_data(data)
    iteration = 1
    while batch_size * iteration < len(data):
        iteration += 1

    return test_images[:batch_size * (iteration - 1)], test_labels[:batch_size * (iteration - 1)]


def get_validation_data(batch_size):
    training_file = h5.File(TRAINING_DIR + 'fruit_data.hdf5', 'r')
    images, labels = training_file["Validation"]["images"], training_file["Validation"]["labels"]
    iteration = 1
    while batch_size * iteration < len(images):
        iteration += 1

    return images[:batch_size * (iteration - 1)], labels[:batch_size * (iteration - 1)]


def format_data(data, outputFile, name):
    shape = (len(data), 100, 100, 3)
    group = outputFile.create_group(name)

    imageSet = group.create_dataset("images", shape, dtype=np.float64)
    labelSet = group.create_dataset("labels", (len(data),), dtype="i")

    random.shuffle(data)

    for index, image_label_pair in enumerate(data):
        imageSet[index, :, :, :] = image_to_array(image_label_pair[0])
        labelSet[index] = image_label_pair[1]
    return imageSet, labelSet