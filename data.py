import cv2
import os
import numpy as np
import random
import pickle
import h5py as h5

train_path = 'fruits/fruits-360/Training/'
test_path =  'fruits/fruits-360/Test/'

CLASSES = ["Apple Braeburn", "Apple Golden 1", "Apple Golden 2", "Apple Golden 3",
            "Apple Granny Smith", "Apple Red 1", "Apple Red 2", "Apple Red 3",
           "Apple Red Delicious", "Apple Red Yellow", "Apricot","Avocado", "Avocado Ripe",  "Banana", "Banana Red", "Cactus Fruit",
            "Carambula", "Cherry", "Clementine", "Cocos","Dates", "Granadilla", "Grape Pink", "Grape White", "Grape White 2",
            "Grapefruit Pink", "Grapefruit White","Guava", "Huckleberry", "Kaki","Kiwi",  "Kumquats", "Lemon", "Lemon Meyer", "Lime", "Lychee", "Mandarin", "Mango", 
            "Maracuja",  "Nectarine", "Orange", "Papaya", "Passion Fruit", "Peach", "Peach Flat", "Pear", "Pear Abate", "Pear Monster", "Pear Williams",
           "Pepino", "Pineapple", "Pitahaya Red", "Plum", "Pomegranate", "Quince", "Raspberry",
            "Salak", "Strawberry", "Tamarillo", "Tangelo"]

def image_to_array(imagePath):
    #Convert an image path to an array that can be used for classification
    return cv2.imread(imagePath)

def get_label(label, one_hot=False):
    if not one_hot:
        return CLASSES.index(label)
    vector = np.zeros(len(CLASSES))
    vector[(CLASSES.index(label))] = 1
    return vector

def get_training_data():
    try:
        training_file = h5.File(TRAINING_DIR+'fruit_data.hdf5', 'r')
        return training_file["Training"]["images"], training_file["Training"]["labels"]
    except:
        print("Constructing training data from scratch")
        paths_and_labels = [] 
        for subdir, dirs, files in os.walk(train_path):
            for file in files:
                label = os.path.basename(subdir)
                if label in CLASSES:
                    path = os.path.join(subdir, file)
                    paths_and_labels.append((path, get_label(label)))

        random.shuffle(paths_and_labels)
        split_index = len(paths_and_labels) // 5

        training_half = paths_and_labels[split_index:]
        validation_half = paths_and_labels[:split_index]

        training_file = h5.File(train_path+'fruit_data.hdf5', 'w')
        
        training_images, training_labels = format_data(training_half, training_file, "Training")
        validation_images, validation_labels = format_data(validation_half, training_file, "Validation")
        
        return training_images, training_labels
    #try:
     #   training_pickle = open(os.path.join(train_path, "training.pkl"), 'rb')
      #  images, labels = pickle.load(training_pickle)
       # print("Loaded training_data from pickle")
        #training_pickle.close()

        #combo = list(zip(images, labels))
        #random.shuffle(combo)
        #images, labels = zip(*combo)

        #return images, labels

def get_test_data(batch_size):
    data = []
    for subdir, dirs, files in os.walk(train_path):
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
    training_file = h5.File(train_path+'fruit_data.hdf5', 'r')
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
        imageSet[index,:,:,:] = image_to_array(image_label_pair[0])
        labelSet[index] = image_label_pair[1]
    return imageSet, labelSet
