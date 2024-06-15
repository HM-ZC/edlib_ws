import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
# 定义网络类
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 类别名称
classes = ['ball', 'other']  # 根据你的实际类别名称

# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net()
model.load_state_dict(torch.load(r'/neural networks/ball_classification_net.pth'))
model = model.to(device)
model.eval()  # 设置为评估模式

# 预处理新图片
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 添加batch维度
    return image

image_path = r'/data1/test/ball/61.jpg'
image = preprocess_image(image_path).to(device)

# 进行预测
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    predicted_class = classes[predicted.item()]

print(f'The predicted class is: {predicted_class}')