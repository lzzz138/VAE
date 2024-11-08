import torch
import torch.optim as optim
from dataset import get_loader
from vae import VAE
from loss import loss_function
from tqdm import tqdm
import numpy as np
from predict import test


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 50
batch_size = 8
learning_rate = 1e-3
input_dim = 784
hidden_dim = 400
z_dim = 20
model = VAE(input_dim, hidden_dim, z_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_loader = get_loader(is_train=True, batch_size=batch_size)
test_loader = get_loader(is_train=False, batch_size=batch_size)
filename = 'result'

if __name__ == '__main__':
    loss_epoch = []
    best_test_loss = np.finfo('f').max
    for epoch in range(num_epochs):
        loop = tqdm(train_loader)
        loss_batch = []
        for images, _ in loop:
            images = images.to(device)
            optimizer.zero_grad()
            outputs, u, log_var = model(images)
            loss, BCE, KLD = loss_function(outputs, images, u, log_var)
            loss.backward()
            optimizer.step()
            loss_batch.append(loss.item())

            loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
            loop.set_postfix(loss=loss.item(), BCE=BCE.item(), KLD=KLD.item())

        loss_epoch.append(np.sum(loss_batch) / len(train_loader.dataset))

        if epoch % 5 == 0:
            best_test_loss = test(model, test_loader, device, z_dim, batch_size, epoch, best_test_loss, filename)