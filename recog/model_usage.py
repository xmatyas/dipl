from tensorflow.keras.models import load_model
import numpy as np
import os
import yaml

def load_config(config_path: str = "config.yaml"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, config_path)
    return yaml.safe_load(open(config_path))

config = load_config()
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), config['MODELS_DIR'])

# Function to load the main model and it's submodels
def load_custom_models(model_name: str = config['MAIN_MODEL'], submodels : list = None):
    main_model = os.path.join(MODELS_DIR + model_name)
    model = load_model(main_model)
    if submodels is not None:
        submodels = [load_model(MODELS_DIR + submodel) for submodel in submodels]
        return model, submodels
    return model

def get_class_label(predicted_class_label, class_map : str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"):
    predicted_class_label = np.argmax(predicted_class_label, axis=1)
    return class_map[predicted_class_label[0]]


def predict_images(images, main_model = None, class_map : str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"):
    if main_model is None:
        main_model = load_custom_models()
    predictions = []
    for sequences in images:
        for image in sequences:
            predicted_class_label = main_model(image)
            predictions.append(get_class_label(predicted_class_label, class_map))
    return predictions

if __name__ == '__main__':
    # It's possible to parse submodels to this as well ['qo_emnist_model', 'vu_emnist_model']
    main_model = load_custom_models(os.environ.get('MODEL_NAME'))