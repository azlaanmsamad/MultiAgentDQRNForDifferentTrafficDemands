import yaml
import logging

def getParameters(filename:str) -> dict:
    """
    @param filename yaml file
    @return: dictionary with parameters in given (yaml) file
    """
    with open(filename, 'r') as stream:
        try:
            parameters = yaml.safe_load(stream)['parameters']
        except yaml.YAMLError as exc:
            logging.error(exc)
    return parameters

