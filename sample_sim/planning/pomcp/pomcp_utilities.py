import json
import logging
import time
from json import JSONDecodeError

def get_default_low_param():
    return 0.0
def get_default_hi_param():
    return 1.0

def get_uct_c(filename, logger_name,objective_function):
    retries = 10
    while retries >= 0:
        try:
            with open(f"params/{objective_function}-{filename.replace('/', '').replace(':','')}.json", "r") as f:
                file = json.load(f)
                return file["low"], file["high"]
        except FileNotFoundError:
            #assert False
            logging.getLogger(logger_name).warning("Params File not found, Returning default")
            return get_default_low_param(), get_default_hi_param()  #extreme high and low numbers seen during running fn:curved env
        except JSONDecodeError:
            logging.getLogger(logger_name).warning("Params File not corrupt, retrying")
            time.sleep(0.1)
        retries -= 1
    #assert False
    logging.getLogger(logger_name).warning("Params file not recoverable, returning defaults")
    return get_default_low_param(), get_default_hi_param()  # extreme high and low numbers seen during running fn:curved env


class PomcpExtraData:
    def __init__(self, objective_c, data_model, filename, state_space_dimensionality,
                 quantiles,objective_function,environment,sensor,grid):
        self.objective_c = objective_c
        self.data_model = data_model
        self.filename = filename
        self.state_space_dimensionality = state_space_dimensionality
        self.quantiles = quantiles
        self.objective_function = objective_function
        self.environment = environment
        self.sensor = sensor
        self.grid = grid

