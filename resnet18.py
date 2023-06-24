import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# 设置随机种子（方便复现）
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 定义一个函数，返回CIFAR10数据集的train和test子集
def get_cifar10():
    # 定义训练集和测试集的数据预处理方式
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),    # 随机裁剪，大小为32x32，填充为4
        transforms.RandomHorizontalFlip(),       # 随机水平翻转
        transforms.ToTensor(),                   # 转换为Tensor对象
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    # 标准化，均值为0.5，标准差为0.5
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),                   # 转换为Tensor对象
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    # 标准化，均值为0.5，标准差为0.5
    ])

    # 加载CIFAR10数据集，定义训练集和测试集（如果没有下载数据集，会自动下载）
    trainset = datasets.CIFAR10(root='D:\PycharmProjects\pythonProject2', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='D:\PycharmProjects\pythonProject2', train=False, download=True, transform=transform_test)

    # 返回训练集和测试集
    return trainset, testset
# 获取CIFAR10数据集的train和test子集

# 定义ResNet-18模型
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=False, num_classes=10)#得到torchvision自带resnet18模型

    def forward(self, x):
        out = self.resnet18(x)
        return out

if __name__ == '__main__':
    trainset, testset = get_cifar10()

    # 定义数据集的dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)#shuffle：是否随机打乱，num_worker同时进程数



    net = ResNet18()

    # 定义损失函数和优化算法
    criterion = nn.CrossEntropyLoss()#crossentropy loss
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    #随机梯度下降（SGD）算法的优化器，用于更新网络权重。 lr=0.1 表示设置学习率为 0.1， momentum=0.9 表示采用动量梯度更新算法,weight_decay=5e-4 表示对网络参数使用L2正则化
    losstrend=[];#方便作图，保存loss
    # 训练模型
    for epoch in range(50):
        running_loss = 0.0  # 初始化loss为0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data  # 获取输入图片及其标签

            optimizer.zero_grad()  # 将梯度缓存设置为0

            outputs = net(inputs)  # 将输入图片经过ResNet网络计算得到预测输出
            loss = criterion(outputs, labels)  # 计算预测输出和实际标签之间的损失函数值
            loss.backward()  # 根据损失函数值计算每个参数需要更新的梯度
            optimizer.step()  # 通过优化器更新网络参数
            running_loss += loss.item()  # 更新总损失

            if i % 100 == 99:  # 每100个batch输出一次训练过程中的平均损失
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                losstrend.append(running_loss / 100)
                running_loss = 0.0  # 训练完一个epoch后将总损失归零
    #保存模型
    # 训练完模型后，保存模型
    PATH = 'D:\PycharmProjects\pythonProject2\pre_resnet18.pth'
    torch.save(net.state_dict(), PATH)
    #可视化
    plt.plot(range(150),losstrend)
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.savefig("losstrend.png")
    plt.close()
    # 测试模型
    correct = 0  # 记录网络分类正确的图片数
    total = 0  # 记录参与分类的图片总数
    with torch.no_grad():  # 给定输入，不会进行梯度计算，在推断时速度更快
        for data in testloader:  # 加载测试数据
            images, labels = data  # 获取输入图片及其标签
            outputs = net(images)  # 将输入图片经过神经网络计算得到预测输出
            _, predicted = torch.max(outputs.data, 1)  # 以每行数据中元素最大的下标作为预测结果
            total += labels.size(0)  # 总共参与预测的图片数量
            correct += (predicted == labels).sum().item()  # 累计图片的分类结果是否正确

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

