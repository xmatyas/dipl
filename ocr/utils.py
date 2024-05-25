import numpy as np
import os
import cv2 as cv
from PIL import Image
import yaml
##########################################################################################
#           Display an image using OpenCV in a proper interactive window                 #
##########################################################################################

def show_image(image : np.ndarray):
    # Check type
    if not isinstance(image, np.ndarray):
        raise TypeError('Image must be a numpy array')
    cv.imshow('Image', image)
    while True:
        # Check if any key is pressed or the window is closed
        key = cv.waitKey(1) & 0xFF
        if key == 27 or cv.getWindowProperty('Image', cv.WND_PROP_VISIBLE) < 1:
            # If ESC key is pressed or the window is closed, exit the loop
            break

    cv.destroyAllWindows()
    return None

##########################################################################################
#           Load and convert an image to grayscale or PIL format                         #
##########################################################################################

def load_and_convert(image, grayscale = True, return_pil = False):
    # Check instance type of image
    if isinstance(image, str):
        #current_dir = os.path.dirname(os.path.abspath(__file__))
        #image = os.path.join(current_dir, image)
        image = cv.imread(image)
        if image is None:
            raise ValueError('Invalid image path')
    elif isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise ValueError('Invalid input type')
    
    # Convert image to grayscale if specified
    if len(image.shape) == 3 and grayscale == True:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Convert image to PIL format if specified
    if return_pil == True:
        return Image.fromarray(image)
    else:
        return image

##########################################################################################
#     [util function]  Validate the arguments for the resize_with_ratio function         #
##########################################################################################

def validate_resize_arguments(width, height, percent):
    if width is None and height is None and percent is None:
        raise ValueError("At least one of 'width', 'height', or 'percent' must be provided")
    if width is not None and width <= 0:
        raise ValueError("Width must be a positive integer")
    if height is not None and height <= 0:
        raise ValueError("Height must be a positive integer")
    if percent is not None and (percent <= 0): # or percent > 100):
        raise ValueError("Percent must be a positive integer less than or equal to 100")

##########################################################################################
#     [util function] Get the size of an image in PIL or numpy format                    #
##########################################################################################

def get_image_size(image):
    if isinstance(image, Image.Image):
        # width, height = image.size
        return image.size 
    elif isinstance(image, np.ndarray):
        # height, width = image.shape[:2]
        return image.shape[1], image.shape[0]
    else:
        raise ValueError("Unsupported image type. Supported types are PIL.Image and numpy.ndarray.")

##########################################################################################
#   [util function] Resize an image with a given width, height, or percentage            #
##########################################################################################

def resize_image(image, size):
    if isinstance(image, Image.Image):
        return image.resize(size, Image.LANCZOS)
    elif isinstance(image, np.ndarray):
        return cv.resize(image, size, interpolation=cv.INTER_LANCZOS4)
    else:
        raise ValueError("Unsupported image type. Supported types are PIL.Image and numpy.ndarray.")

##########################################################################################
#   [main function] Resize an image with a given width, height, or percentage            #
##########################################################################################

def resize_with_ratio(image, width : int = None, height : int = None, percent : int = None, grayscale : bool = True, return_pil : bool = True) -> Image.Image:
    # Convert the image to PIL format
    image = load_and_convert(image, grayscale = grayscale, return_pil = return_pil)
    # Validate the arguments
    validate_resize_arguments(width, height, percent)
    
    # Calculate the aspect ratio
    original_width, original_height = get_image_size(image)
    aspect_ratio = original_height / original_width
    
    if percent is not None:
        width = int(original_width * (percent / 100))
        height = int(original_height * (percent / 100))
    elif width is None:
        width = int(height / aspect_ratio)
    elif height is None:
        height = int(width * aspect_ratio)
    
    # Resize the image
    image = resize_image(image, (width, height))
    
    # Return the resized image (parse the input arguments to the function)
    return image

##########################################################################################
#   [util function] Get the configuration from the YAML file                             #
##########################################################################################

def load_config(config_file : str = 'config.yaml') -> dict:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(current_dir, config_file)
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    if config is None:
        raise ValueError('Invalid configuration file / Empty configuration file / Not found')
    return config