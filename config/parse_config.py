import yaml
import os

def parse_config(name) :
    class ConfigClass :
        def __init__(self, **entries) :
            for key, value in entries.items() :
                if isinstance(value, dict) :
                    value = ConfigClass(**value)
                self.__dict__.update({key : value})
    with open(f'./config/{name}.yaml', encoding='utf-8') as f :
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = ConfigClass(**config)
    return config