import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


# alexnet resize(224)
# googlenet resize(96)
# cifar-10 image size is [10, 3, 32, 32]
# FashionMNIST [10, 1, 28, 28]
def getTrainDataLoader(batch_size, resize=None):
    train_set = torchvision.datasets.FashionMNIST(
        root='datas',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize)
        ])
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader

def getTestDataLoader(batch_size, resize=None):
    # testing data
    test_set = torchvision.datasets.FashionMNIST(
        root='datas',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize)
        ])
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True
    )
    return test_loader

