import yaml
from flask import Flask

# Load configuration from config.yaml
def load_config(flask_app: Flask, config_file_path: str) -> int:
    try:
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
            flask_app.config.update(config)
    except FileNotFoundError:
        print("Config file not found")
    except yaml.YAMLError as e:
        print("Error loading config file:", e)
    except Exception as e:
        print("An error occurred:", e)
    return 0