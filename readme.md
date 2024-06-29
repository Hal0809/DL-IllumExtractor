# 基于深度学习的图像色彩信息提取算法及图像颜色还原
by Hal  

**本文提到的 RVFL 非真正意义上的 RVFL，只是一个拙劣的模仿！模型源码见arch文件夹下。**

~~Q：为什么用 RVFL？  
A：来自于那什么都不懂的”导师“提出的荒谬想法。我知道这毫无科学性,且是彻头彻尾的胡闹。若将来有类似文章发表与本人无关！~~
## 主要结果对比展示
###### 若图片无法加载请使用加速器
###### 基于DenseNet201模型的结果
![基于DenseNet201模型的结果](results%2Ftest_top5%2F0%2Fcopm_res.png)

###### 基于DenseNet201-CBAM模型的结果
![基于DenseNet-CBAM模型的结果](results%2Ftest_top5%2F1%2Fcopm_res.png)

###### 基于DenseNet201-CBAM-RVFL'模型的结果
![基于DenseNet-CBAM-RVFL‘的结果](results%2Ftest_top5%2F3%2Fcopm_res.png)  
图片上下两张为一组，分别为使用深度学习算法得到的图片和真实（Ground truth）图片。

这是我的本科毕业设计项目。该项目的主要功能是实现了将RAW格式的图像转换为RGB色彩空间下的彩色图像。

## 数据集
本项目使用 [SimpleCube++](https://github.com/Visillect/CubePlusPlus) 数据集。
以下为来自数据集的原始图片：
![SC++.png](results%2FSC%2B%2B.png)
训练集目录结构：  
data  
+-- SimpleCube++  
----+-- test  
--------+-- PNG  
--------+-- gt.csv  
----+--train  
--------+-- PNG  
--------+--gt.csv


## 代码
### Pytorch
#### 主要需求
1. Python 3.8
2. pytorch 
3. cudatoolkit
4. numpy
5. Pillow
6. matplotlib
7. pandas
8. opencv-python

### 训练
运行 'Den_Train.py', 'Den_CBAM_Train.py', 'Den_RVFL_Train.py', 'Den_CBAM_RVFL_Train.py' 
训练得到4个模型。其中 Den_RVFL 模型几乎不会成功，后续不用于进行图片生成。Den_CBAM_RVFL 模型也有较大可能失败。训练结果参数保存在 models 文件夹下。
训练过程数据保存在 results 文件夹下，以 'den' 开头的 csv 文件。

### 图片生成
运行 'utility.py'，可以对测试集中的50张图片进行特征提取和色彩生成。首先对测试集中的50张图片进行特征提取，得到的结果保存至 results 文件夹下，
以 'eval' 开头的 csv 文件。随后从中找到误差最大和最小的图片各5张，结果保存在以 'res' 开头的 csv 文件中。最后对误差最小的5张图片进行色彩生成，结果保存在 test_top5 文件夹中。

## 结果
见上。模型训练数据，包括 MSE Loss 数据见 'den_xxxx.csv' 文件，位于 results 文件夹下。

本项目主要应用了深度学习网络DenseNet201进行图像色彩信息特征提取，并对其进行了简单改写以适应本文的回归任务的需求。  
###### 深度学习部分的所有内容由我一人独立完成（再次痛斥我的所谓”导师“在过程中没有提供任何实质性指导的行为），图像还原部分（仅指 get_preview()函数及被其调用的内容）来自互联网。  
本项目仅供研究学习。如果有问题或疑问，可以联系我：Gqh123123@qq.com  
=_=（2024.6.27）
