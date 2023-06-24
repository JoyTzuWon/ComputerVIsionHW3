import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
#losstrend=[0.591,0.039,0.182,0.140,0.003,0.005,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.051,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.001,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000]
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
if __name__ == '__main__':
    trainset, testset = get_cifar10()
    resnet18 = models.resnet18(pretrained=False)
    resnet18.fc = nn.Identity()
    resnet18.load_state_dict(torch.load('D:\PycharmProjects\pythonProject2\self_resnet18.pth'))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)#shuffle：是否随机打乱，num_worker同时进程数

    num_epochs=50
    num_classes = 10
    losstrend=[]
# Replace the last layer
    in_features = 512
    classifier = nn.Linear(in_features, num_classes)
    resnet18.fc = classifier
    for param in resnet18.parameters():
        param.requires_grad = False
    for param in resnet18.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet18.fc.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            optimizer.zero_grad()

        # Forward pass
            outputs = resnet18(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                    (epoch+1, i+1, running_loss/100))
                losstrend.append(running_loss/100)
                running_loss = 0.0
    PATH = 'D:\PycharmProjects\pythonProject2\\finetune_resnet18.pth'
    torch.save(resnet18.state_dict(), PATH)
    #可视化
    plt.plot(range(150),losstrend)
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.savefig("losstrend2.png")
    plt.close()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = resnet18(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {:.2f}%'.format(100 * correct / total))
'''
# Define a new classification model that uses the trained resnet18 model as a feature extractor
class ClassificationModel(nn.Module):
    def __init__(self, feature_extractor, num_classes):
        super(ClassificationModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

num_classes = 10
classification_model = ClassificationModel(resnet18, num_classes)


# Evaluate the performance of the classification model on the test dataset
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = classification_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: {:.2f}%'.format(100 * correct / total))
'''
