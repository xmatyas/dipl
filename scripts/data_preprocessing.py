# %% [markdown]
# # Data parsing
# Parse data from its binary form (idx) and then flip and normalise it

# %%
import numpy as np
import struct
import os

def validate_path(path: str):
    if path is None:
        return False
    return os.path.exists(path)

def get_idx_type(path):
    magic_number = int.from_bytes(open(path, 'rb').read(4), byteorder='big')
    magic_number_mapping = {
        0x00000801: 1,
        0x00000803: 3
    }
    return magic_number_mapping.get(magic_number, "Unknown")

def normalize_images(data):
    return data / 255.0

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

def parse_binary_file(path, debug=False):
    magic_number = get_idx_type(path)
    with open(path, 'rb') as f:
        _, size = struct.unpack('>II', f.read(8))
        if magic_number == 1:
            data = np.fromfile(f, dtype=np.uint8).newbyteorder('>')
            data = data.reshape(size)
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
            # data = reshape_data(data)
            if debug:
                print(f"Magic number: {magic_number}")
                print(f"Size: {size}")
                print(f"Height: {nrows} px")
                print(f"Width: {ncols} px")
            return data
        else:
            print(f"Unknown file type: {magic_number}")
            return None

def filter_images(images : np.array, labels : np.array, class_map: str, filter_labels : str) :
    filter_labels = set(filter_labels)
    if len(filter_labels) == 0:
        return images, labels
    filtered_data = [(img, label) for img, label in zip(images, labels) if str(class_map[label]) in filter_labels]
    filtered_images, filtered_labels = zip(*filtered_data)
    filtered_images = np.array(filtered_images)
    filtered_labels = np.array(filtered_labels)
    return filtered_images, filtered_labels
    

train_images = parse_binary_file("../input/emnist_source_files/emnist-bymerge-train-images-idx3-ubyte")
train_labels = parse_binary_file("../input/emnist_source_files/emnist-bymerge-train-labels-idx1-ubyte")
train_unique_labels = get_unique_labels(path="../input/emnist_source_files/emnist-bymerge-train-labels-idx1-ubyte")
test_images = parse_binary_file("../input/emnist_source_files/emnist-bymerge-test-images-idx3-ubyte")
test_labels = parse_binary_file("../input/emnist_source_files/emnist-bymerge-test-labels-idx1-ubyte")
test_unique_labels = get_unique_labels(path="../input/emnist_source_files/emnist-bymerge-test-labels-idx1-ubyte")
only_o_and_q_images, only_o_and_q_labels = filter_images(train_images, train_labels, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt","UV")
print("Images : {} | Labels : {}".format(only_o_and_q_images.shape, only_o_and_q_labels.shape))


# %% [markdown]
# # Setting the GPU parameters

# %%
def setTensorflowGPUParams():
    from os import environ
    environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    environ['KERAS_BACKEND'] = 'tensorflow'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    #tf.debugging.set_log_device_placement(True)
    tf.test.is_gpu_available()
    devices = tf.config.experimental.list_physical_devices('GPU')
    GPU = devices[0]
    try:
        tf.config.experimental.set_memory_growth(GPU, True)
    except:
        pass
setTensorflowGPUParams()

# %%
import matplotlib.pyplot as plt
print(test_images.shape)
class_mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
number_of_classes = len(class_mapping)
index_number = 12345
plt.imshow(train_images[index_number], cmap='gray')
plt.title(f"Label: {class_mapping[train_labels[index_number]]}")
plt.show()

# %%



