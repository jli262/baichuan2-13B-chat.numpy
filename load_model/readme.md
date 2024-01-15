## load_model.py说明  

直接根据百川bin模型文件，把模型加载到内存中，大约消耗内存40GB左右，小内存机器勿轻易尝试  

提供小工具类，根据layer name把每个权重读取出来，pytorh的tensor对象及相互转化函数  

重点说明：  
```
在baichuan2 13B chat中，bin文件使用的是bfloat16精度，而读取的float为float32位精度  

16bit -> 32bit的过程中存在后16位的精度胡编乱造  

如 1.3244 -> 1.324410000  

但是bfloat16是转为谷歌TPU设备而生的数据格式，对于CPU和GPU用户来说没什么用，也没有什么加速效果  

```