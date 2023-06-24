import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import random
import matplotlib.pyplot as plt

class TripletCIFAR10():
    def __init__(self, cifar10_dataset):
        self.cifar10_dataset = cifar10_dataset
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        anchor_img, _ = self.cifar10_dataset[index]
        positive_img = self.transform(anchor_img)

        negative_index = random.randint(0, len(self.cifar10_dataset) - 1)
        negative_img, _ = self.cifar10_dataset[negative_index]

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.cifar10_dataset)


# 加载CIFAR10数据集并对图像进行预处理

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
cifar10_dataset = datasets.CIFAR10(root='D:\PycharmProjects\pythonProject2\data', train=True, transform=transform, download=True)
triplet_cifar10_dataset = TripletCIFAR10(cifar10_dataset)
train_loader = torch.utils.data.DataLoader(triplet_cifar10_dataset, batch_size=32, shuffle=True)
# 定义ResNet18网络，并通过它提取出每个图像的特征向量
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Identity()

# 定义对比损失函数
criterion = nn.TripletMarginLoss()

# 定义优化器
optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)
losstrend=[]
# 开始训练
for epoch in range(10):
    for i, (anchor, positive, negative) in enumerate(train_loader):
        optimizer.zero_grad()
        anchor_embedding = resnet18(anchor)
        positive_embedding = resnet18(positive)
        negative_embedding = resnet18(negative)
        loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            losstrend.append(loss.item())
            print("Epoch {}, iteration {}, loss = {:.3f}".format(epoch, i, loss.item()))

PATH = 'D:\PycharmProjects\pythonProject2\self_resnet18.pth'
torch.save(resnet18.state_dict(), PATH)
plt.plot(range(160),losstrend)
plt.xlabel("epoch")
plt.ylabel("train loss")
plt.savefig("losstrend1.png")
plt.close()
# 测试模型性能
test_dataset = datasets.CIFAR10(root='D:\PycharmProjects\pythonProject2\data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = resnet18(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: {:.2f}%'.format(100 * correct / total))
