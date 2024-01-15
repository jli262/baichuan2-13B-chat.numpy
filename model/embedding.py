import numpy as np


class Embedding:
    def __init__(self, model_path, vocab_size, embedding_dim, padding_idx):
        # Initialize parameters
        self.weights = np.load(model_path)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        padding_idx = self.vocab_size + padding_idx
        self.padding_idx = padding_idx

    def forward(self, input_ids):
        print("embedding.inputsids", input_ids.shape, " ", input_ids.dtype)
        print("embedding.weights", self.weights.ndim)
        if input_ids.ndim == 1:
            return self.weights[input_ids]

        size = list(input_ids.shape) + list(self.weights.shape[1:])

        x = self.weights[input_ids.ravel()].reshape(size)
        print("xxxxxxxxxxx", x.ndim)
        return x

