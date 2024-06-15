import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# 类别名称
classes = ['ball', 'other']  # 根据你的实际类别名称

# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(classes))
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

image_path = r'/data1/test/ball/62.jpg'
image = preprocess_image(image_path).to(device)

# 进行预测
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    predicted_class = classes[predicted.item()]

print(f'The predicted class is: {predicted_class}')