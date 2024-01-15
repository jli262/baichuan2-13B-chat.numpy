import numpy as np
from .embedding import Embedding
from .RMSNorm import RMSNorm
from typing import Tuple, Optional, List, Union
from .alibi_mask import _gen_alibi_mask
from .baichuanlayer import BaichuanLayer


class BaseModelOutputWithPast:
    def __init__(self, last_hidden_state, past_key_values=None, hidden_states=None, attentions=None):
        """
        :param last_hidden_state: Output of the last layer of the model.
        :param past_key_values: (Optional) List of past key value states.
        :param hidden_states: (Optional) List of hidden states from each layer.
        :param attentions: (Optional) List of attention weights from each layer.
        """
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions


class BaichuanModel:
    def __init__(self, model_path, config):
        self.config = config
        self.padding_idx = config["pad_token_id"]
        self.vocab_size = config["vocab_size"]
        self.n_head = config["num_attention_heads"]
        # 定义embedding 初始化状态（未执行）
        # config.vocab_size : 词典的大小尺寸，比如总共出现5000个词，那就输入5000
        # config.hidden_size : 嵌入向量的维度，即用多少维来表示一个符号（单词）
        # padding_idx : 填充id，比如，输入长度为100，但是每次的句子长度并不一样，后面就需要用统一的数字填充，而这里就是指定这个数字，这样，网络在遇到填充id时，就不会计算其与其它符号的相关性。（初始化为0）
        self.embed_tokens = Embedding(model_path+"/model.embed_tokens.weight.npy", config["vocab_size"],
                                      config["hidden_size"], self.padding_idx)
        self.norm = RMSNorm(model_path + "/model.norm.weight.npy", config["hidden_size"], epsilon=config["rms_norm_eps"])
        self.gradient_checkpointing = config["gradient_checkpointing"]
        # 初始化transformer模型权重
        self.max_cache_pos = config["model_max_length"]
        self.first_run = True
        self.alibi_mask = None
        self.buffers = {}
        self.layers = [BaichuanLayer(model_path+"/model.layers." + str(i), self.config) for i in range(self.config["num_hidden_layers"])]

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def register_buffer(self, attr, content, persistent=True):
        self.buffers[attr] = content

    def get_alibi_mask(self, input, seq_length_with_past):
        if self.first_run:
            self.first_run = False
            future_mask = _gen_alibi_mask(input, self.n_head, self.max_cache_pos)
            self.register_buffer(
                "future_mask",
                future_mask,
                persistent=False,
            )
        if seq_length_with_past > self.max_cache_pos:
            self.max_cache_pos = seq_length_with_past
            self.register_buffer(
                "future_mask",
                _gen_alibi_mask(input, self.n_head, self.max_cache_pos),
                persistent=False,
            )
        mask = self.buffers["future_mask"][
               : self.n_head, :seq_length_with_past, :seq_length_with_past
               ]

        return mask

    def forward(self,
                input_ids=None,
                attention_mask: Optional[np.array] = None,
                past_key_values: Optional[List[np.array]] = None,
                inputs_embeds: Optional[np.array] = None,
                use_cache: Optional[bool] = False,
                output_attentions: Optional[bool] = False,
                output_hidden_states: Optional[bool] = False,
                return_dict: Optional[bool] = True,
                ) -> Union[Tuple, BaseModelOutputWithPast]:

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot provide both input_ids and inputs_embeds simultaneously"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You need to provide input_ids or inputs_embeds")

        return_dict = (
            return_dict if return_dict is not None else self.config["use_return_dict"]
        )

        seq_length_with_past = seq_length

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens.forward(input_ids)

        alibi_mask = self.get_alibi_mask(inputs_embeds, seq_length_with_past)

        if attention_mask is not None:
            print("model.alibi_mask", alibi_mask.ndim)
            print("1111111111111")
            if len(attention_mask.shape) == 2:
                expanded_mask = attention_mask.astype(alibi_mask.dtype)
                expanded_mask = np.tril(
                    np.greater(expanded_mask[:, :, None] * expanded_mask[:, None, :], 0)
                ) * np.equal(expanded_mask[:, :, None] - expanded_mask[:, None, :], 0)

            else:
                expanded_mask = attention_mask
            bsz = inputs_embeds.shape(0)
            src_len, tgt_len = alibi_mask.shape()[-2:]
            expanded_mask = (
                np.broadcast_to(
                    np.expand_dims(expanded_mask, 1), (bsz, 1, src_len, tgt_len)
                ).astype(alibi_mask.dtype)
            )
            inverted_mask = 1.0 - expanded_mask

            bool_mask = inverted_mask.astype(bool)
            inverted_mask = np.where(bool_mask, np.finfo(inverted_mask.dtype).min, inverted_mask)

            attention_mask = inverted_mask + np.expand_dims(alibi_mask, 0)
        else:
            attention_mask = alibi_mask
            print("model.alibi_mask", alibi_mask.ndim)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )
            print("model.hiddenstates", hidden_states.shape)
            layer_outputs = decoder_layer.forward(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm.forward(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )

        # 最终模型输出
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
