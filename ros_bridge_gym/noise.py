import numpy as np

from stable_baselines3.common.noise import ActionNoise, NormalActionNoise


class RandomActionNoise(ActionNoise):

    def __init__(self, shape, scale=1., epsilon=.9, epsilon_decay=.99995):
        super().__init__()
        self._shape = shape
        self._scale = scale
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay


    def __call__(self):
        self._epsilon *= self._epsilon_decay
        if np.random.random() < self._epsilon:
            return (np.random.random(self._shape) - 0.5) * self._scale
        else:
            return np.zeros(self._shape)


class EpsilonNormalActionNoise(NormalActionNoise):

    def __init__(self, mean, sigma, epsilon=.9, epsilon_decay=.99995):
        super().__init__(mean, sigma)
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay


    def __call__(self):
        noise = super().__call__()
        self._epsilon *= self._epsilon_decay
        if np.random.random() < self._epsilon:
            return noise
        else:
            return np.zeros(self._mu.shape)
    