import numpy
import torch
import struct
import os

"""
    获取百川模型文件，把模型加载到内存中
    转化出来的数据为bfloat16 -> fp32 对于非TPU用户来说，bfloat16没有实际意义
    所以转化出来的实际模型可能比3个bin加起来要大
    将拿到的numpy weight以numpy数据文件的格式，按weight name存在numpy_weight中
"""


class Model:

    def __init__(self, model_path):
        self.__model_path = model_path
        self.__model_weights = {}
        self.__load_model()

    # 加载模型
    def __load_model(self):
        all_files = os.listdir(self.__model_path)
        model_files = [file for file in all_files if file.endswith(".bin")]
        for file in model_files:
            file = os.path.abspath(self.__model_path + '/' + file)
            model_state_dict = torch.load(file, map_location='cpu')
            for layer_name, weights in model_state_dict.items():
                self.__model_weights[layer_name] = weights.float().numpy()

    # 获取具体layer name的权重
    def get_layer(self, name: str):
        return self.__model_weights[name]

    # tensor -> numpy (fp32) 存在精度损失
    @staticmethod
    def tensor2numpy(tensor: torch.Tensor) -> numpy.ndarray:
        return tensor.float().numpy()

    # 数据持久化到本地 float32
    @staticmethod
    def save_tensor(data: numpy.ndarray, output: str):
        shape = data.shape
        data = data.flatten()
        with open(output, 'wb+') as f:
            for d in data:
                f.write(struct.pack('f', d))

    # 读取持久化的一个float32数据
    @staticmethod
    def read_a_fp32(file: str) -> float:
        with open(file, 'rb') as f:
            bin = f.read(4)
            data = struct.unpack('f', bin)[0]
        return data

    def getWeight(self):
        return self.__model_weights


if __name__ == '__main__':
    m = Model("/opt/infertest/baichuan2_cpu/_numpy/models")
    # 按layer保存numpy model weights
    for name, array in m.getWeight().items():
        numpy.save(f'/opt/infertest/baichuan2_cpu/_numpy/numpy_weight/{name}.npy', array)
    # numpy model weights全部保存进一个npz文件
    # numpy.savez_compressed('numpy_weights.npz', **m.getWeight())
