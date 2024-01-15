import numpy as np
from .attention import BaichuanAttention
from .MLP import MLP
from .RMSNorm import RMSNorm
from typing import Optional, Tuple


class BaichuanLayer:
    def __init__(self, model_path, config):
        self.hidden_size = config['hidden_size']
        self.self_attn = BaichuanAttention(model_path, config=config)
        self.mlp = MLP(
            model_path+".mlp",
            hidden_size=self.hidden_size,
            intermediate_size=config["intermediate_size"],
            hidden_act=config["hidden_act"]
        )
        self.input_layernorm = RMSNorm(model_path + ".input_layernorm.weight.npy", config["hidden_size"], epsilon=config["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(
            model_path + ".post_attention_layernorm.weight.npy", config["hidden_size"], epsilon=config["rms_norm_eps"]
        )

    def forward(
            self,
            hidden_states: np.array,
            attention_mask: Optional[np.array] = None,
            past_key_value: Optional[Tuple[np.array]] = None,
            output_attentions: Optional[bool] = None,
            use_cache: Optional[bool] = False,
    ) -> Tuple[
        np.array, Optional[Tuple[np.array, np.array]]
    ]:

        residual = hidden_states
        print("layer.hiddenstates", hidden_states.shape)

        hidden_states = self.input_layernorm.forward(hidden_states)
        print("layer.hiddenstates", hidden_states.shape)
        # 百川 Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn.forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
