import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models

if __name__ == '__main__':
    # 检查GPU是否可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # 定义数据变换，包括数据增强
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载训练和测试数据集
    train_dataset = ImageFolder(root='C:\\Users\\14168\\1\\Python\\pythonProject\\data1\\train', transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    test_dataset = ImageFolder(root='C:\\Users\\14168\\1\\Python\\pythonProject\\data1\\test', transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 类别名称
    classes = train_dataset.classes
    num_classes = len(classes)
    print(classes)

    # 使用预训练的ResNet18模型
    net = models.resnet18(pretrained=True)

    # 替换最后的全连接层，适应当前的分类任务
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, num_classes)

    net = net.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    # 训练模型
    num_epochs = 20
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Iter {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # 测试模型
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct / total}%')

    # 保存模型
    torch.save(net.state_dict(), '/neural networks/ball_classification_net.pth')