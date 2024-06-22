import numpy as np

from ..State import State
class ElasticFEMState(State):

    def __init__(self, x, x_prev):
        self.x = x
        self.x_prev = x_prev
        return

    def primary(self):
        return self.x


    # def mixed(self):
    #     return np.zeros((0, 1))