import os

def get_data_directory():
    return os.path.join(os.path.dirname(__file__), "../../", "data/")