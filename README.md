# TextCNN
使用TextCNN完成中文新闻分类，数据集为THUCNews

数据集：THUCNews，数据集来源：[数据集下载](https://github.com/649453932/Chinese-Text-Classification-Pytorch/tree/master/THUCNews/data)

包含：数据处理，模型，训练，推理，各部分代码下载即可使用。

# 使用
1. 从[数据集下载](https://github.com/649453932/Chinese-Text-Classification-Pytorch/tree/master/THUCNews/data)下载数据集，放到data文件夹下
2. 根据提示修改`save_vocab.py`, `main.py`, `inference.py`, `train.py`中的数据集/模型保存/模型加载 路径
3. 训练：执行`python train.py`
4. 推理：执行`python inference.py`


# 说明
## 数据处理
数据预处理：`save_data.py`: 读取数据，分词，保存数据

数据集：`dataset.py`: 继承自`torch.utils.data.Dataset`，用于加载数据集

## 模型
TextCNN：`model.py`：TextCNN模型

## 训练
训练：`train.py`：训练模型
  
## 推理
推理：`inference.py`：加载模型，并进行预测

  