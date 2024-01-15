import sys
sys.path.append('_numpy/')
#include <sentencepiece_processor.h> in c++
import os
import sentencepiece as spm

def getTokenizer(tokenizer_path):

    # 实际上 tokenization_utils_base中的from_pretrained处理的就是下面的 vocab_files
    vocab_files = {
        'vocab_file': os.path.abspath(tokenizer_path + '/tokenizer.model'),
        'added_tokens_file': None,
        'special_tokens_map_file': os.path.abspath(tokenizer_path + 'special_tokens_map.json'),
        'tokenizer_config_file': os.path.abspath(tokenizer_path + 'tokenizer_config.json')
    }

    tokenizer_model = vocab_files.get('vocab_file')
    sp_model = spm.SentencePieceProcessor(model_file=tokenizer_model)
    return sp_model
    # 如果需要从model文件读取二进制转换，需要BPE算法解码，这个后期再搞
    # 在百川2的词典内 有BPE编码 和少量复杂汉字的utf-8编码，所以 单纯的byte.decode('utf-8')并不可行
    # -----------------------------------------------------------------------------
