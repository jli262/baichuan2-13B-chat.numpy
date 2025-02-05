"""
baichuan2-chat 13B 的numpy推理
"""
import sys
import os
from model.tokenizer import getTokenizer
import json
from model.baichuanforcausallm import BaichuanForCausalLM
import time

MODEL_PATH = "/opt/infertest/baichuan2_cpu/_numpy/models"
CONFIG_path = "/opt/infertest/baichuan2_cpu/_numpy/models/config.json"
GENERATION_CONFIG_PATH = "/opt/infertest/baichuan2_cpu/_numpy/models/generation_config.json"
SPECIAL_TOKENS_MAP = "/opt/infertest/baichuan2_cpu/_numpy/models/special_tokens_map.json"
WEIGHT_PATH = "/opt/infertest/baichuan2_cpu/_numpy/numpy_weight/"


def run():
    # 从tokenizer.tokenizer处拿到sentencepiece的tokenizer model
    tokenizer = getTokenizer(tokenizer_path=os.path.join(MODEL_PATH))

    print('tokenizer装载模型 ok. > ', tokenizer)

    # 直接读取config的json文件，储存为dict()格式
    with open(CONFIG_path) as config_json:
        config = json.load(config_json)
    with open(GENERATION_CONFIG_PATH) as generation_config_json:
        generation_config = json.load(generation_config_json)

    print('chat config file 装载 ok. > ', generation_config)

    start = time.time()
    # 本地版本
    model_path = os.path.abspath(WEIGHT_PATH)
    model = BaichuanForCausalLM(model_path, config)
    end = time.time()
    print('Baichuan2 CausalLM装载模型 ok. > ', end - start)

    messages = []
    messages.append({"role": "user", "content": "请直接结束对话。"})

    """
    "介绍一下你自己"
    ['<reserved_106>介绍一下你自己<reserved_107>我是。。。']
    """

    response = model.chat(tokenizer, messages, True, config, generation_config)

    print(response)

if __name__ == '__main__':
    run()
