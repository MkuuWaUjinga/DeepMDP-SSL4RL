import numpy as np

from garage.np.baselines.base import Baseline

# Mod of garage's zero baseline implementation with that doesn't offer predict_n function as vpg algo errors with it

class ZeroBaseline(Baseline):

    def __init__(self, env_spec):
        pass

    def get_param_values(self, **kwargs):
        return None

    def set_param_values(self, val, **kwargs):
        pass

    def fit(self, paths):
        pass

    def predict(self, path):
        return np.zeros_like(path['rewards'])
