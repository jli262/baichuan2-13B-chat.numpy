import numpy as np
from .dense import Dense
from typing import Optional, Tuple
import math
from .activations import Softmax


class BaichuanAttention:
    def __init__(self, model_path, config):
        self.config = config
        self.hidden_size = config["hidden_size"]
        # 注意力头数
        self.num_heads = config["num_attention_heads"]
        # 每个注意力头的维度
        self.head_dim = self.hidden_size // self.num_heads
        # 模型的最大长度
        self.max_position_embeddings = config["model_max_length"]

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by num_heads {self.num_heads}"
            )

        # 先声明一个Linear类，并没有实例化 (只是标记了该layer的形状)
        self.W_pack = Dense(
            model_path + ".self_attn.W_pack.weight.npy", self.hidden_size, 3 * self.hidden_size, use_bias=False
        )
        # 跟上面一样
        self.o_proj = Dense(
            model_path + ".self_attn.o_proj.weight.npy", self.num_heads * self.head_dim, self.hidden_size, use_bias=False
        )

    def forward(self, hidden_states: np.array,
                attention_mask: Optional[np.array] = None,
                past_key_value: Optional[Tuple[np.array]] = None,
                output_attentions: bool = False,
                use_cache: bool = False
                ) -> Tuple[np.array, Optional[np.array], Optional[Tuple[np.array]]]:
        print("attention.hiddenstates", hidden_states.ndim)
        if attention_mask is not None:
            print("attention.attention_mask", attention_mask.ndim)
        if past_key_value is not None:
            print("attention.past_key_value", len(past_key_value))
        bsz, q_len, _ = hidden_states.shape
        print("attention.hidden_states", hidden_states.shape)
        proj = self.W_pack.forward(hidden_states)
        print("attention.proj", proj.shape)
        # Step 1: Reshape the last dimension
        # Step 2: Add a new dimension at the beginning
        # Step 3: Transpose the first and the second-to-last dimensions
        # Step 4: Remove the second-to-last dimension if it's of size 1
        proj = (
            np.squeeze(
                np.swapaxes(
                    np.expand_dims(
                        proj.reshape(*proj.shape[:-1], 3, self.hidden_size), 0), 0, -2), axis=-2)
        )
        print("attention.proj", proj.shape)
        query_states = (
            proj[0].reshape(
                bsz, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        )
        print("attention.q", query_states.shape)
        key_states = (
            proj[1].reshape(
                bsz, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        )

        value_states = (
            proj[2].reshape(
                bsz, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        )

        # Tensor 倒数第二个维度的大小
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

            # reuse k, v, self_attention
            # 拼接张量 --> 按照我的理解 就是如果不是第一层，就把上一层的输出和这一层的k v 拼起来
            key_states = np.concatenate([past_key_value[0], key_states], axis=2)
            value_states = np.concatenate([past_key_value[1], value_states], axis=2)

        # 如果不用缓存 实在没搞懂这个操作
        # 参考：https://zhuanlan.zhihu.com/p/628511161?utm_id=0
        # 参考：https://zhuanlan.zhihu.com/p/632229531
        # 作用：用于推理加速 默认不开启
        past_key_value = (key_states, value_states) if use_cache else None

        # 矩阵乘法运算 再除以 head_dim(注意力头维度)的非负平方根
        attn_weights = np.matmul(
            query_states, key_states.transpose(0, 1, 3, 2)
        ) / math.sqrt(self.head_dim)

        print("q, k", query_states.shape, key_states.transpose(0, 1, 3, 2).shape)

        print("attention_weights", attn_weights.ndim, attn_weights.shape)
        # 注意力机制掩码 盲猜用于decoder入口
        if attention_mask is not None:
            if q_len == 1:  # inference with cache
                if len(attention_mask.size()) == 4:
                    attention_mask = attention_mask[:, :, -1:, :]
                else:
                    attention_mask = attention_mask[:, -1:, :]
            print("33333333333333", attn_weights.shape, attention_mask.shape)
            attn_weights = attn_weights + attention_mask
            attn_weights = np.maximum(attn_weights, np.finfo(attn_weights.dtype).min)
        print("attention_mask", attention_mask.ndim)
        attn_weights = Softmax(attn_weights)
        attn_output = np.matmul(attn_weights, value_states)
        print("we, k, o", attn_weights.shape, value_states.shape, attn_output.shape)
        attn_output = attn_output.transpose(0, 2, 1, 3)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj.forward(attn_output)

        if not output_attentions:
            attn_weights = None
        print("attn_output", attn_output.ndim)
        # 返回注意力输出、注意力权重和过去的键值对
        return attn_output, attn_weights, past_key_value
