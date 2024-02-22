import numpy as np
import struct
import matplotlib.pyplot as plt
import json
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with open('../input/emnist_source_files/emnist-bymerge-train-images-idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    train_data = train_data.reshape((size,nrows,ncols))
print('Train images shape : ',np.shape(train_data))
with open('../input/emnist_source_files/emnist-bymerge-test-images-idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    test_data = test_data.reshape((size,nrows,ncols))
print('Test images shape : ',np.shape(test_data))
with open('../input/emnist_source_files/emnist-bymerge-train-labels-idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    train_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    train_labels = train_labels.reshape((size,))
print('Train labels shape : ',np.shape(train_labels))
with open('../input/emnist_source_files/emnist-bymerge-test-labels-idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    test_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    test_labels = test_labels.reshape((size,))
print('Test labels shape : ',np.shape(test_labels))

class_map = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
number_of_classes = len(class_map)
print('Class size : ', number_of_classes)

# Data normalisation
train_data = train_data / 255.0
test_data = test_data / 255.0

train_data_size = train_data.shape[0]
train_data_height = 28
train_data_width = 28
train_data_img_size = train_data_height*train_data_width

train_data = train_data.reshape(train_data_size, train_data_height, train_data_width, 1)

test_data_size = test_data.shape[0]
test_data_height = 28
test_data_width = 28
test_data_img_size = test_data_height*test_data_width

test_data = test_data.reshape(test_data_size, test_data_height, test_data_width, 1)

# Transform labels
train_labels = to_categorical(train_labels, number_of_classes)
test_labels = to_categorical(test_labels, number_of_classes)
# Split some data for validation
train_data, validation_data, train_labels, validation_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

print("###### Final shapes ######")
print("Train data shape : ", np.shape(train_data), "\tTrain labels shape : ", np.shape(train_labels))
print("Test data shape : ", np.shape(test_data), "\t\tTest labels shape : ", np.shape(test_labels))
print("Validation data shape : ", np.shape(validation_data), "\tValidation labels shape : ", np.shape(validation_labels))

devices = tf.config.experimental.list_physical_devices('GPU')
device = devices[0]
tf.config.experimental.set_memory_growth(device, True)
# Convolution-subsampling pairs
# https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist/notebook#1.-How-many-convolution-subsambling-pairs?
nets = 3
model = [0] *nets
names = ["(C-P)x1","(C-P)x2","(C-P)x3"]

for j in range(nets):
    model[j] = Sequential()
    model[j].add(Conv2D(24, kernel_size=5, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model[j].add(MaxPooling2D())
    if j>0:
        model[j].add(Conv2D(48, kernel_size=5, padding='same', activation='relu'))
        model[j].add(MaxPooling2D())
    if j>1:
        model[j].add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))
        model[j].add(MaxPooling2D(padding='same'))
    model[j].add(Flatten())
    model[j].add(Dense(256, activation='relu'))
    model[j].add(Dense(number_of_classes, activation='softmax'))
    model[j].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = [0] * nets
epochs = 20
batch_size = 128
for j in range(nets):
    history[j] = model[j].fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(validation_data, validation_labels), verbose=0)
    print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(names[j],epochs,max(history[j].history['accuracy']),max(history[j].history['val_accuracy']) ))
    
# Feature maps
nets = 6
model = [0] *nets
for j in range(6):
    model[j] = Sequential()
    model[j].add(Conv2D(j*8+8,kernel_size=5,activation='relu',input_shape=(28,28,1)))
    model[j].add(MaxPooling2D())
    model[j].add(Conv2D(j*16+16,kernel_size=5,activation='relu'))
    model[j].add(MaxPooling2D())
    model[j].add(Flatten())
    model[j].add(Dense(256, activation='relu'))
    model[j].add(Dense(number_of_classes, activation='softmax'))
    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
history = [0] * nets
names = ["8 maps","16 maps","24 maps","32 maps","48 maps","64 maps"]
epochs = 20
batch_size = 128
for j in range(nets):
    history[j] = model[j].fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(validation_data, validation_labels), verbose=0)
    print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(names[j],epochs,max(history[j].history['accuracy']),max(history[j].history['val_accuracy']) ))
    
model = tf.keras.Sequential([
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=24, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(filters=48, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(padding='same'),
    tf.keras.layers.Flatten(),
    #################################################
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(number_of_classes, activation='softmax')
])
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer= opt,loss='categorical_crossentropy',metrics=tf.keras.metrics.CategoricalAccuracy())

es = EarlyStopping(monitor='val_loss', mode='max', verbose=1, patience=10)
mc = ModelCheckpoint('combined_emnist_model.h5', monitor='val_categorical_accuracy', mode='max', verbose=1, save_best_only=True, patience=5)
history = model.fit(train_data, train_labels, batch_size=256 , epochs=100, validation_data= (validation_data,validation_labels) , callbacks=[es,mc], verbose=1)
with open('history.json', 'w') as f:
    json.dump(history.history, f)

model = load_model('combined_emnist_model.h5')

def evaluate(testX, testY, model):
    scores, histories = list(), list()
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    scores.append(acc)
    histories.append(history)
    return scores, histories

def summarize_diagnostics(histories):
    for index in range(len(histories)):
        # LOSS
        plt.subplot(2,1,1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[index].history['loss'], color='blue', label='train')
        plt.plot(histories[index].history['val_loss'], color='orange', label='test')
        # ACCURACY
        plt.subplot(2,2,2)
        plt.title('Classification Categorical Accuracy')
        plt.plot(histories[index].history['categorical_accuracy'], color='blue', label='train')
        plt.plot(histories[index].history['val_categorical_accuracy'], color='orange', label='test')
        plt.show()

def summarize_performance(scores):
    print('Accuracy: mean=%.3f std=%.3f n=%d' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))

model = load_model('combined_emnist_model.h5')
scores, histories = evaluate(test_data, test_labels, model)
summarize_diagnostics(histories)
summarize_performance(scores)