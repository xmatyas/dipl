# Importing os is needed because importing only 'path' produces some naming conflicts with the 'path' variable in the function
import os
import struct
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from set_params import set_tensorflow_gpu_params
set_tensorflow_gpu_params()

# Function to validate the path (Just a helper function)
def validate_path(path: str):
    if path is None:
        return Exception("Path is not provided.")
    if not os.path.exists(path):
        return Exception("Path does not exist.")
    return True
# Function to get the idx type of given file from path
def get_idx_type(path):
    magic_number = int.from_bytes(open(path, 'rb').read(4), byteorder='big')
    magic_number_mapping = {
        0x00000801: 1,
        0x00000803: 3
    }
    return magic_number_mapping.get(magic_number, "Unknown")
# Function to normalize the images to 0-1 range
def normalize_images(data):
    return data / 255.0
# Function to reshape the data to 2D
def reshape_data(data):
    print('Previous shape : {}'.format(data.shape))
    if len(data.shape) != 3:
        return Exception("Data is not in the correct shape")
    size = data.shape[0]
    height = data.shape[1]
    width = data.shape[2]
    data = data.reshape(size, height * width)
    print('New shape : {}'.format(data.shape))
    return data
# Function to get the unique labels from the labels array or from the file
def get_unique_labels(labels : np.ndarray = None, path : str = None):
    if validate_path(path) == False:
        print("Path does not exist.")
        return None
    if path is not None:
        if get_idx_type(path) != 1:
            print("Wrong idx type detected. Please use idx1 files.")
            return None
        with open(path, 'rb') as f:
            header = f.read(8)
            labels = set(f.read())
    elif labels is not None:
        labels = set(labels)
        return len(labels)
    else:
        print("No labels provided.")
        return None        
# Function to create a class map from the mapping file
def create_class_map(path : str = "../input/emnist-bymerge-mapping.txt"):
    validate_path(path)
    class_map = {}
    with open(path) as f:
        for line in f:
            (key, val) = line.split()
            class_map[int(key)] = chr(int(val))
    return class_map
# Function to parse the binary file and return the data
def parse_binary_file(path, debug=False):
    magic_number = get_idx_type(path)
    with open(path, 'rb') as f:
        _, size = struct.unpack('>II', f.read(8))
        if magic_number == 1:
            data = np.fromfile(f, dtype=np.uint8).newbyteorder('>')
            data = data.reshape(size)
            class_map = create_class_map()
            number_of_classes = len(class_map)
            data = to_categorical(data, number_of_classes)
            if debug:
                print(f"Magic number: {magic_number}")
                print(f"Size: {size}")
            return data
        elif magic_number == 3:
            nrows, ncols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.uint8).newbyteorder('>')
            # Reshapes to Images x Height x Width
            data = data.reshape(size, nrows, ncols)
            # Flip images so they are right side up
            data = np.transpose(data, (0, 2, 1))
            # Normalize images
            data = normalize_images(data)
            # Reshape to Images x (Height * Width)
            #data = reshape_data(data)
            if debug:
                print(f"Magic number: {magic_number}")
                print(f"Size: {size}")
                print(f"Height: {nrows} px")
                print(f"Width: {ncols} px")
            return data
        else:
            print(f"Unknown file type: {magic_number}")
            return None
def get_class_number_from_categorized_label(label : np.ndarray):
    index = np.where(label == 1.0)[0][0]
    return index    
# Function to filter the images based on the labels
def filter_images(images : np.array, labels : np.array, class_map: str, filter_labels : str, reverse: bool = False) :
    filter_labels = set(filter_labels)
    if len(filter_labels) == 0:
        return images, labels
    if reverse:
        filtered_data = [(img, label) for img, label in zip(images, labels) if str(class_map[get_class_number_from_categorized_label(label)]) in filter_labels]
    else:
        filtered_data = [(img, label) for img, label in zip(images, labels) if str(class_map[get_class_number_from_categorized_label(label)]) not in filter_labels]
    filtered_images, filtered_labels = zip(*filtered_data)
    filtered_images = np.array(filtered_images)
    filtered_labels = np.array(filtered_labels)
    return filtered_images, filtered_labels

def get_confusion_matrix(model, test_images, test_labels):
    predicted_labels = model.predict(test_images)
    predicted_labels = np.argmax(predicted_labels, axis=1)
    test_labels = np.argmax(test_labels, axis=1)
    cm = confusion_matrix(test_labels, predicted_labels)
    print(cm)
    return cm

train_images = parse_binary_file("../input/emnist_source_files/emnist-bymerge-train-images-idx3-ubyte")
train_labels = parse_binary_file("../input/emnist_source_files/emnist-bymerge-train-labels-idx1-ubyte")
test_images = parse_binary_file("../input/emnist_source_files/emnist-bymerge-test-images-idx3-ubyte")
test_labels = parse_binary_file("../input/emnist_source_files/emnist-bymerge-test-labels-idx1-ubyte")
print("Train images : {} | Train labels : {}".format(train_images.shape, train_labels.shape))
print("Test images : {} | Test labels : {}".format(test_images.shape, test_labels.shape))

class_map = create_class_map()
print("Class map : {}".format(class_map))
number_of_classes = len(class_map)
train_unique_labels = get_unique_labels(path="../input/emnist_source_files/emnist-bymerge-train-labels-idx1-ubyte")
test_unique_labels = get_unique_labels(path="../input/emnist_source_files/emnist-bymerge-test-labels-idx1-ubyte")
train_images, train_labels = filter_images(train_images, train_labels, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt","QOUV", reverse = False)
test_images, test_labels = filter_images(test_images, test_labels, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt","QOUV", reverse = False)
number_of_classes = number_of_classes - len("QOUV") # remove the filtered labels
print("Images : {} | Labels : {}".format(train_images.shape, train_labels.shape))

def create_model(number_of_classes: int ,nets: int = 2, maps: int = 24, density: tuple = (128,64)) -> Sequential:
    model = Sequential()
    for net_layer in range(nets):
        model.add(Conv2D(net_layer*maps+maps, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    for density_layer in density:
        model.add(Dense(density_layer, activation='relu'))
        model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy','categorical_crossentropy', 'top_k_categorical_accuracy'])
    return model

def train_model(model: Sequential, train_images: np.array, train_labels: np.array, test_images: np.array, test_labels: np.array, epochs: int = 10):
    mc = ModelCheckpoint('new_emnist_model.h5', monitor='val_categorical_crossentropy', mode='max', verbose=1, save_best_only=True)  
    model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=epochs, batch_size=256, callbacks=mc, verbose=1)
    return model

model = create_model(number_of_classes, 2, 32, (256,128,64))
print(model.summary())
train_model(model, train_images, train_labels, test_images, test_labels)