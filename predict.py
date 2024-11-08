import shutil
import torch
from torchvision.utils import save_image
from loss import loss_function
import os


def save_checkpoint(state, is_best, filename):
    if not os.path.exists(filename):
        os.mkdir(filename)
    checkpoint_path = os.path.join(filename, 'checkpoint.pth')
    best_path = os.path.join(filename, 'best.pth')
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)


def test(model, test_loader, device, z_dim, batch_size, epoch, best_test_loss, filename):
    test_avg_loss = 0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            output, u, log_var = model(images)
            test_loss, BCE, KLD = loss_function(output, images, u, log_var)
            test_avg_loss += test_loss.item()
        test_avg_loss /= len(test_loader.dataset)

        # 随机生成正态分布利用decoder生成图片
        z = torch.randn(batch_size, z_dim).to(device)
        output = model.decode(z).view(batch_size, 1, 28, 28)
        save_image(output, f'generate_imgs/sampled_{epoch+1}.png')

        # 保存现有的模型
        is_best = test_avg_loss < best_test_loss

        if best_test_loss > test_avg_loss:
            best_test_loss = test_avg_loss

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_test_loss': best_test_loss,
        }, is_best, filename)

        return best_test_loss