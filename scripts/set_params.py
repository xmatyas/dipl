from os import environ
import tensorflow as tf

# Set GPU params - this is needed for proper GPU memory allocation and to avoid OOM errors
def set_tensorflow_gpu_params():
    environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    environ['KERAS_BACKEND'] = 'tensorflow'
    environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    #tf.get_logger().setLevel('ERROR')
    #tf.debugging.set_log_device_placement(True)
    #tf.test.is_gpu_available()
    devices = tf.config.experimental.list_physical_devices('GPU')
    GPU = devices[0]
    try:
        tf.config.experimental.set_memory_growth(GPU, True)
    except:
        pass