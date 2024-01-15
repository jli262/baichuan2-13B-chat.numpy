from .activations import SiLU
from .dense import Dense
from .dropout import Dropout


class MLP:
    def __init__(self, model_path, hidden_size=512, intermediate_size=2048, hidden_act="silu"):
        self.gate_proj = Dense(model_path+".gate_proj.weight.npy", inputs_num=hidden_size, units_num=intermediate_size)
        self.down_proj = Dense(model_path+".down_proj.weight.npy", inputs_num=intermediate_size, units_num=intermediate_size)
        self.up_proj = Dense(model_path+".up_proj.weight.npy", inputs_num=hidden_size, units_num=intermediate_size)
        self.activation = SiLU

    def forward(self, x):
        print("mlp.x", x.ndim)
        return self.down_proj.forward(self.activation(self.gate_proj.forward(x)) * self.up_proj.forward(x))
