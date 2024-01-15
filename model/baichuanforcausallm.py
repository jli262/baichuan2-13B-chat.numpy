import time
from threading import Thread
from typing import List, Optional, Tuple, Union

import numpy as np
from utils.generation_utils import build_chat_input, TextIterStreamer

from .BaichuanModel import BaichuanModel
from .activations import Softmax


class CausalLMOutputWithPast:
    def __init__(self, loss, logits, past_key_values, hidden_states, attentions):
        self.loss = loss
        self.logits = logits
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions


class NormHead:
    def __init__(self, model_path, hidden_size, vocab_size, bias=False):
        self.weight = np.load(model_path)

        self.first_flag = True

    def forward(self, hidden_states):
        norm_weight = self.weight
        print("norm weight", norm_weight.ndim)
        return np.dot(hidden_states, norm_weight.T)


class BaichuanForCausalLM:
    def __init__(self, model_path, config):
        self.model = BaichuanModel(model_path, config)
        self.lm_head = NormHead(model_path + "/lm_head.weight.npy", config["hidden_size"], config["vocab_size"], bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def prepare_inputs_for_generation(
                self,
                input_ids: np.array,
                past_key_values: Optional[np.array] = None,
                attention_mask: Optional[np.array] = None,
                inputs_embeds: Optional[np.array] = None,
                **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )

        return model_inputs

    def forward(
            self,
            input_ids: np.array = None,
            attention_mask: Optional[np.array] = None,
            past_key_values: Optional[List[np.array]] = None,
            inputs_embeds: Optional[np.array] = None,
            labels: Optional[np.array] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state

        print("forward.hidden_states", hidden_states.ndim, hidden_states.shape)
        logits = self.lm_head.forward(hidden_states)
        loss = None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def RepetitionPenaltyLogitsProcessor(self, input_ids, logits, penalty):
        ranges = np.arange(logits.shape[0])
        logit = logits[ranges[:, None], input_ids]
        score = np.where(logit < 0, logit * penalty, logit / penalty)

        logits[ranges[:, None], input_ids] = score

        return logits

    def TemperatureLogitsWarper(self, input_ids, logits, temperature):
        logits = logits / temperature
        return logits

    def TopPLogitsWarper(self, input_ids, logits, top_p, filter_value=-float("Inf"), min_tokens_to_keep=1):
        sorted_indices = np.argsort(-logits)  # -scores for descending order
        sorted_logits = np.sort(-logits)  # Sort in ascending order and negate to get descending
        sorted_logits = -sorted_logits  # Negate again to get the original values in descending order
        softmax_probs = Softmax(sorted_logits)
        cumulative_probs = np.cumsum(softmax_probs, axis=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0

        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].copy()
        sorted_indices_to_remove[..., 0] = 0
        # scatter sorted tensors to original indexing
        # Initialize an array to store the results
        indices_to_remove = np.zeros_like(sorted_indices_to_remove)

        # Iterate over the first dimension to simulate scatter behavior
        for i in range(sorted_indices_to_remove.shape[0]):
            indices_to_remove[i, sorted_indices[i]] = sorted_indices_to_remove[
                i, np.arange(sorted_indices_to_remove.shape[1])]

        logits[indices_to_remove] = filter_value
        return logits

    def TopKLogitsWarper(self, input_ids, scores, top_k, filter_value=-float("Inf"), min_tokens_to_keep=1):
        top_k = min(max(top_k, min_tokens_to_keep), scores.shape[-1])
        top_k_scores = np.partition(-scores, top_k - 1, axis=-1)[..., :top_k]
        threshold = top_k_scores[..., -1, None]
        indices_to_remove = scores < -threshold
        scores[indices_to_remove] = filter_value
        return scores

    def sample(
            self,
            input_ids: np.array,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: bool = False,
            streamer: Optional = None,
            **model_kwargs,
    ):
        # stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        # "model_max_length"做一个stopping criteria
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        # RepetitionPenaltyLogitsProcessor需要加上

        # TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper需要加上

        scores = None
        decoder_attentions = None
        cross_attentions = None
        decoder_hidden_states = None

        # keep track of which sequences are already finished
        unfinished_sequences = np.ones(input_ids.shape[0], dtype=np.int64)

        while True:
            # prepare model inputs
            print("???????????/", input_ids.shape)
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            print("!!!!!!!!!!!!!", model_inputs["input_ids"].shape)
            start = time.time()
            # forward pass to get next token
            outputs = self.forward(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            print("sample.output", outputs.logits.ndim)
            print()
            end = time.time()
            print("Time for one forward(): ", end - start)
            # Process the new logits
            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = self.RepetitionPenaltyLogitsProcessor(input_ids, next_token_logits, self.generation_config["repetition_penalty"])
            next_token_scores = self.TemperatureLogitsWarper(input_ids, next_token_scores, self.generation_config["temperature"])
            next_token_scores = self.TopPLogitsWarper(input_ids, next_token_scores, self.generation_config["top_p"])
            next_token_scores = self.TopKLogitsWarper(input_ids, next_token_scores, self.generation_config["top_k"])

            # 没有定义return_dict_in_generate，应该是没有用到的
            # if return_dict_in_generate:
            #     if output_scores:
            #         scores += (next_token_scores,)
            #     if output_attentions:
            #         decoder_attentions += (
            #             (outputs.attentions,)
            #         )
            #     if output_hidden_states:
            #         decoder_hidden_states += (
            #             (outputs.decoder_hidden_states,)
            #         )

            probs = Softmax(next_token_scores, -1)
            # 报错，改后代码有待验证
            # next_tokens = np.random.choice(probs.shape[1], size=(probs.shape[0], 1), p=probs)
            next_tokens = np.array([np.random.choice(len(probs[i]), 1, p=probs[i]) for i in range(len(probs))]).squeeze(1)

            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = np.concatenate([input_ids, next_tokens[:, None]], axis=-1)

            if streamer is not None:
                streamer.put(next_tokens)
            """
            作用不知道，先不管
            model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            """

            cur_len = input_ids.shape[-1]
            this_peer_finished = (cur_len >= self.config["model_max_length"]) or (next_tokens[0] == eos_token_id)
            if this_peer_finished and not synced_gpus:
                break

            print(input_ids)

        if streamer is not None:
            streamer.end()

        return input_ids

    def generate(self, inputs, config, generation_config, streamer=None):
            self.config = config
            self.generation_config = generation_config

            eos_token_id = generation_config["eos_token_id"]
            generation_config["pad_token_id"] = eos_token_id

            batch_size = inputs.shape[0]
            print("generate.input_ids", inputs.shape)
            input_ids = inputs
            if streamer is not None:
                streamer.put(input_ids)

            input_ids_length = input_ids.shape[-1]

            generation_config["max_length"] = generation_config["max_new_tokens"] + input_ids_length

            generation_mode = "SAMPLE"

            #记得做一个maxlength stop处理

            # _expand_inputs_for_generation
            input_ids = np.repeat(input_ids, repeats=1, axis=0)
            print("generate.input_ids", input_ids.shape)
            return self.sample(
                input_ids,
                pad_token_id=generation_config["pad_token_id"],
                eos_token_id=generation_config["eos_token_id"],
                streamer=streamer,
                config=self.config,
            )

    def chat(self, tokenizer, messages: List[dict], stream=False, config: Optional = None,
                generation_config: Optional = None):
        self.generation_config = generation_config
        # input_ids 实际上是一个torch 张量， 函数内已经包含了message encode
        input_ids = build_chat_input(config, generation_config, tokenizer, messages, generation_config["max_new_tokens"])
        print("chat.input_ids", input_ids.ndim)
        # 流式输出 实际上python是个迭代器
        if stream:
            streamer = TextIterStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            Thread(target=self.generate, kwargs=dict(
                inputs=input_ids, streamer=streamer,
                config=config,
                generation_config=generation_config,
            )).start()
            return streamer
        else:
            # 执行forward -> 代码在：Transformers库 /generation/utils.py中
            outputs = self.generate(input_ids, config, generation_config=generation_config)
            # 输出结果解码
            response = tokenizer.decode(outputs[0][len(input_ids[0]):])
            return response
