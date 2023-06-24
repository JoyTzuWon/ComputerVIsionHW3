# ComputerVIsionHW3
作业1对比监督学习与自监督学习在CIFAR-10或CIFAR-100图像分类任务中的性能表现


数据集选择：CIFAR-10


数据集介绍：数据集由6万张32*32的彩色图片组成，一共有10个类别。每个类别6000张图片。其中有5万张训练图片及1万张测试图片。数据集被划分为5个训练块和1个测试块，每个块1万张图片。测试块包含了1000张从每个类别中随机选择的图片。训练块包含随机的剩余图像，但某些训练块可能对于一个类别的包含多于其他类别，训练块包含来自各个类别的5000张图片。这些类是完全互斥的，及在一个类别中出现的图片不会出现在其它类中。


Part1 基于resnet18作用在CIFAR-10数据集上的监督学习


骨干网络：resnet18


参数选择 ：batch_size=128,num_workers=2(num_worker同时进程数),epoch=50,优化器选择：随机梯度下降（SGD）算法的优化器，用于更新网络权重。 lr=0.1 表示设置学习率为 0.1， momentum=0.9 表示采用动量梯度更新算法，weight_decay=5e-4表示对网络参数使用L2正则化


训练示例：


<img width="349" alt="image" src="https://github.com/JoyTzuWon/ComputerVIsionHW3/assets/129930916/f3f878df-8860-4c36-8b4b-cff388849148">


Loss曲线迭代：




<img width="415" alt="image" src="https://github.com/JoyTzuWon/ComputerVIsionHW3/assets/129930916/6e272caf-6c1d-469a-8d95-ece871d164ef">






Linear Classification Protocol：




<img width="416" alt="image" src="https://github.com/JoyTzuWon/ComputerVIsionHW3/assets/129930916/fa9d4067-8b4d-46da-9787-ff9915efe365">








准确率为74%。


模型名称：pre_resnet18.pth


Part2 基于resnet18在CIFAR-10数据集上的自监督学习


自监督学习方法：对比学习


参数选择：参数选择 batch_size=32,num_workers=2(num_worker同时进程数), epoch=10,优化器选择：随机梯度下降（SGD）算法的优化器，用于更新网络权重。 lr=0.1 表示设置学习率为 0.1， momentum=0.9 表示采用动量梯度更新算法。图片数据增强：transforms.RandomCrop(32, padding=4)  # 随机裁剪，大小为32x32，填充为4


transforms.RandomHorizontalFlip() # 随机水平翻转


transforms.ToTensor()  # 转换为Tensor对象


transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # 标准化，均值为0.5，标准差为0.5


由于数据增强复杂度与电脑设备条件限制，自监督效果一般，自监督阶段训练loss：





<img width="416" alt="image" src="https://github.com/JoyTzuWon/ComputerVIsionHW3/assets/129930916/519b4e4c-3a63-44d3-a712-51a759373882">







自监督训练阶段模型：self_resnet18.pth


微调阶段训练loss：







<img width="367" alt="image" src="https://github.com/JoyTzuWon/ComputerVIsionHW3/assets/129930916/66c185c2-f99a-4bdb-aca6-8a527c3eeb9e">











Linear Classification Protocol：








<img width="408" alt="image" src="https://github.com/JoyTzuWon/ComputerVIsionHW3/assets/129930916/b3cfab30-1f54-45b8-a1f6-9d8f2ce502e4">










微调过后准确率：38.67%；出现这种自监督效果与监督训练效果有区别的原因可能是由于batchsize，learningrate，训练epoch等参数的选择导致训练结果不佳，由于数据增强比较复杂，训练时间很长，不得不选择较低的batchsize。


微调过后模型：finetune_resnet18.pth


实验一百度网盘地址：https://pan.baidu.com/s/1_SRvdZLsJGFCOFchZ8Lkhg 


提取码：562e 


实验二 使用具有泛化能力的 NeRF 模型，自己构建物体数据集（如手机拍摄），对物体进行三维重建


Nerf模型选择：instant-ngp模型


数据集选择：


Flower重构数据集



<img width="415" alt="image" src="https://github.com/JoyTzuWon/ComputerVIsionHW3/assets/129930916/376ce152-35f2-4e2e-9498-c6845147eb1c">











Fox重构数据集



<img width="365" alt="image" src="https://github.com/JoyTzuWon/ComputerVIsionHW3/assets/129930916/1a8957e8-7209-44d3-bcb9-1ed615ae60f6">










Step 2 Colmap重建相机参数


Flower图片相机参数：






<img width="416" alt="image" src="https://github.com/JoyTzuWon/ComputerVIsionHW3/assets/129930916/239b070e-4318-4dab-b40f-ab677ab8700e">






Fox图片相机参数：不同模型的相机参数会产生transform.json文件




<img width="415" alt="image" src="https://github.com/JoyTzuWon/ComputerVIsionHW3/assets/129930916/d90ee057-b15a-4483-becd-5f005732049d">











图片transform矩阵：







<img width="415" alt="image" src="https://github.com/JoyTzuWon/ComputerVIsionHW3/assets/129930916/4a116cf8-3906-4b49-afae-3537f557dfea">








Step 3 运行模型重建结果：



<img width="415" alt="image" src="https://github.com/JoyTzuWon/ComputerVIsionHW3/assets/129930916/303c4b9b-dc72-43c4-9724-ccc81842b83d">




由于设备显卡处理能力，在重构flower和fox时出现了显存不足CUDA-OUT-OF-MEMORY的问题


尝试运行重建2维图片：






<img width="415" alt="image" src="https://github.com/JoyTzuWon/ComputerVIsionHW3/assets/129930916/564be1df-a460-4749-97ab-7f00d78a010a">








说明instant-ngp完全可以使用，更换先进设备可以完整重构flower&fox模型。


百度网盘模型链接：https://pan.baidu.com/s/1eONwrhRz26y01knDEw0GVw 


提取码：94p6
