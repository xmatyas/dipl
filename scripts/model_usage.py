from tensorflow.keras.models import load_model
import os

os.environ['MODEL_FOLDER'] = '../models/'

# Function to load the main model and it's submodels
def load_custom_models(model_name: str, submodels: list = None):
    if os.environ.get('MODEL_FOLDER') is None:
        raise Exception("MODEL_FOLDER environment variable is not set.")
    else:
        model_folder = os.environ.get('MODEL_FOLDER')
    model = load_model(model_folder + model_name)
    if submodels is not None:
        submodels = [load_model(model_folder + submodel) for submodel in submodels]
        return model, submodels
    return model

main_model, submodels = load_custom_models('emnist_model', ['qo_emnist_model', 'vu_emnist_model'])

import numpy as np
# Test image, this is 784 pixels of a 28x28 image of a number '9'
test_image = 
#test_image = test_image.reshape(1, 28, 28)
#test_image = test_image / 255.0


predicted_class_label = main_model(test_image)
predicted_class_label = np.argmax(predicted_class_label, axis=1)
print(predicted_class_label)


