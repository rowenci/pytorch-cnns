import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("model/trained_models/VGG/trained_vgg9.pth")
model.eval()
batch_size = 10
test_set = torchvision.datasets.CIFAR10(
    root='datas',
    train=False,
    download=False,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False
)

for data in test_loader:
    imgs, labels = data
    imgs, labels = imgs.to(device), labels.to(device)
    outputs = model(imgs)
    print(outputs.argmax(1))
    print(labels)
    print((outputs.argmax(1) == labels).sum())
    break





"""
1. LeNet model
model = LeNet()
"""

"""
2. AlexNet model
model = AlexNet()
"""

"""
3. VGG
ratio = 8
conv_arch = ((1, 1, 64 // ratio), (1, 64 // ratio, 128 // ratio), (2, 128 // ratio, 256 // ratio), (2, 256 // ratio, 512 // ratio), (2, 512 // ratio, 512 // ratio))
fc_features = 512 * 7 * 7
fc_hidden_units = 4096
model = model.VGG.vgg(conv_arch, fc_features // ratio, fc_hidden_units // ratio)
"""

"""
4. NiN
model = model.NiN.get_nin(1, 10)
writer = SummaryWriter("tensorLogs/nin")
"""

"""
5. GoogLeNet
model = model.GoogLeNet.getNet()
writer = SummaryWriter("tensorLogs/googlenet")
"""

"""
model = model.ResNet.getResNet()
"""