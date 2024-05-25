# Thesis Project

This repository contains the code for my personal thesis project. It consists of several modules, including `api`, `ocr`, and `recog`, which are described below.

## Modules

### api

The `api` module provides a RESTful API for interacting with the thesis project. It handles requests and responses, and provides endpoints for various functionalities.

### ocr

The `ocr` module is responsible for optical character recognition (OCR). It takes an image as input and extracts text from it using machine learning algorithms. This module is crucial for the project's text analysis capabilities.

### recog

The `recog` module focuses on object recognition. It uses computer vision techniques to identify and classify objects in images. This module plays a key role in the project's image analysis functionalities.

## Running the Project

To run the project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/xmatyas/dipl.git
    ```

2. Navigate to the project directory:

    ```bash
    cd dipl
    ```

3. Install the required dependencies onto a conda env:

    ```bash
    conda env create -f environment.yml
    ```

4. Run the `main.py` script:

    ```bash
    python main.py
    ```

    OR a better solution would be to use gunicorn

    ```bash
    gunicorn main:flask_app -b 127.0.0.1:5000
    ```

5. Run celery

    ```bash
    celery -A main.celery_app worker
    ```

    This script serves as the entry point for the thesis project. It orchestrates the different modules and executes the main functionality.

That's it! You should now have the project up and running.

## License

This project is licensed under the [MIT License](LICENSE).