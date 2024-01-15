import numpy as np

class Dense:
    """
    Add Dense layer
    ---------------
        Args:
            `units_num` (int): number of neurons in the layer
            `use_bias` (bool):  `True` if used. `False` if not used
        Returns:
            output: data with shape (batch_size, units_num)
    """

    def __init__(self, model_path, units_num, inputs_num=None, use_bias=False):
        self.units_num = units_num
        self.inputs_num = inputs_num
        self.use_bias = use_bias

        self.w = np.load(model_path)
        self.b = None
        self.cache = dict(x=None)

    def forward(self, X):
        print("dense.X", X.ndim)
        print("dense.weights", self.w.ndim)
        z = np.dot(X, self.w.T)
        if self.b is not None:
            z += self.b.data
        self.cache = dict(x=X)

        return z
