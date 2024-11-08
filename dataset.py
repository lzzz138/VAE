from torchvision import datasets, transforms
import torch.utils.data as data


def get_loader(is_train, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if is_train:
        dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
