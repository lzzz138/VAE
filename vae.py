import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, z_dim=20):
        super(VAE, self).__init__()
        # encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, z_dim)
        self.fc3 = nn.Linear(hidden_dim, z_dim)

        # decoder
        self.fc4 = nn.Linear(z_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        # 学习输入的均值u和方差的对数
        x = F.relu(self.fc1(x))
        u = self.fc2(x)
        log_var = self.fc3(x)
        return u, log_var

    def reparameterization(self, u, log_var):
        # 从标准正态分布中采样
        sigma = torch.exp(0.5*log_var)
        eps = torch.randn_like(sigma)
        return eps * sigma + u

    def decode(self, x):
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        u, log_var = self.encode(x)
        z = self.reparameterization(u, log_var)
        x = self.decode(z)
        x_hat = x.view(batch_size, 1, 28, 28)
        return x_hat, u, log_var


if __name__ == '__main__':
    model = VAE()
    a=torch.randn(8,784)
    b,c,d=model(a)
    print(b.shape)