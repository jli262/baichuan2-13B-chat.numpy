import sys
import torch
sys.path.append('_numpy/')
#include <sentencepiece_processor.h> in c++
import os
from typing import List
import tokenization_baichuan
import numpy as np


messages = [{"role": "user", "content": "为什么庞奇辉要蹲着尿尿，很显然他是个男的呀"}]


# 原官方版本
def baichuan_version(messages):
    # 百川tokenizer模型路径
    tokenizer_path = os.path.abspath(r'D:\LLM\Baichuan2_chat_13B_model')
    tokenizer = tokenization_baichuan.BaichuanTokenizer.from_pretrained(tokenizer_path,
                                                                        use_fast=False, trust_remote_code=False)

    # 去除model参数的build_chat_input函数，原函数在generation_utils.py中
    def build_chat_input(tokenizer, messages: List[dict]):
        def _parse_messages(messages, split_role="user"):
            system, rounds = "", []
            round = []
            for i, message in enumerate(messages):
                if message["role"] == "system":
                    assert i == 0
                    system = message["content"]
                    continue
                if message["role"] == split_role and round:
                    rounds.append(round)
                    round = []
                round.append(message)
            if round:
                rounds.append(round)
            return system, rounds

        max_new_tokens = 2048
        max_input_tokens = 2048
        # 解析 切割 message
        system, rounds = _parse_messages(messages, split_role="user")
        # encode编码
        system_tokens = tokenizer.encode(system)  # out : []
        max_history_tokens = max_input_tokens - len(system_tokens)  # out 2048
        # out -> system == "" ; rounds == [[{'role': 'user', 'content': '为什么庞奇辉要蹲着尿尿，很显然他是个男的呀'}]]
        # rounds[::-1] == [[{'role': 'user', 'content': '为什么庞奇辉要蹲着尿尿，很显然他是个男的呀'}]]
        history_tokens = []
        for round in rounds[::-1]:
            round_tokens = []
            for message in round:
                if message["role"] == "user":
                    round_tokens.append(195)
                else:
                    round_tokens.append(196)
                print('transformers库 encode 结果:', tokenizer.encode(message["content"]))
                round_tokens.extend(tokenizer.encode(message["content"]))
            if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
                history_tokens = round_tokens + history_tokens  # concat left
                if len(history_tokens) < max_history_tokens:
                    continue
            break

        input_tokens = system_tokens + history_tokens
        if messages[-1]["role"] != "assistant":
            input_tokens.append(196)
        input_tokens = input_tokens[-max_input_tokens:]  # truncate left

        # 把token转化成张量
        return torch.LongTensor([input_tokens]).to('cpu')

    return build_chat_input(tokenizer, messages)


# numpy版本  cpp重构tokenizer可参考下列代码
def numpy_version(messages):
    tokenizer_path = r'D:\LLM\Baichuan2_chat_13B_model\tokenizer.model'
    # -----------------------------------------------------------------------------
    # sentencepiece  库 使用swig编译成python的，这里转Cpp反而好转 (貌似是做了切词功能)
    import sentencepiece as spm
    sp_model = spm.SentencePieceProcessor(model_file=tokenizer_path)
    # 如果需要从model文件读取二进制转换，需要BPE算法解码，这个后期再搞
    # 在百川2的词典内 有BPE编码 和少量复杂汉字的utf-8编码，所以 单纯的byte.decode('utf-8')并不可行
    # -----------------------------------------------------------------------------

    history_tokens, system_tokens = [], []
    max_history_tokens = 2048
    max_input_tokens = 2048
    rounds = [messages]

    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message["role"] == "user":
                round_tokens.append(195)
            else:
                round_tokens.append(196)
            print('numpy版本直接 encode 结果(int):', sp_model.encode(message["content"], out_type=int))
            round_tokens.extend(sp_model.encode(message["content"]))
        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(196)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left

    # 把token转化成numpy array 对比transformers版本的张量输出
    return np.array([input_tokens])


if __name__ == '__main__':
    baichuan_res = baichuan_version(messages)
    print('百川2 transformers库和torch库 tokenizer 计算结果：')
    print(baichuan_res)
    print('----------------------------------------------')
    print('百川2 mumpy tokenizer 计算结果：')
    numpy_res = numpy_version(messages)
    print(numpy_res)

