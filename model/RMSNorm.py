import numpy as np


class RMSNorm:
    def __init__(self, model_path, hidden_size=None, epsilon=1e-6):
        self.weights = np.load(model_path)

        self.hidden_size = hidden_size
        self.epsilon = epsilon

    def forward(self, x):
        print("rmsnorm.x", x.ndim)
        print("rmsnorm.weights", self.weights.ndim)
        variance = np.mean(np.square(x.astype(np.float32)), axis=-1, keepdims=True)

        x = x * np.reciprocal(np.sqrt(variance + self.epsilon))

        return self.weights * x
