from model.tokenizer import getTokenizer
import os
MODEL_PATH = "/opt/infertest/baichuan2_cpu/_numpy/models"

tokenizer = getTokenizer(tokenizer_path=os.path.join(MODEL_PATH))

output = [[94723, 34, 104521, 73, 73, 73, 73, 73
        ]]

output = tokenizer.decode(output)
print(output)
