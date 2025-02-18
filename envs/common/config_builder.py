import yaml

class Configuration:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, Configuration(**value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return Configuration(**config_data)
