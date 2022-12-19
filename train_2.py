import numpy as np
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import dataloader
import model_3dunet
import model_fsrcnn

import os
import skimage.io as io


import matplotlib.pyplot as plt

if __name__ == '__main__':
    # device = torch.device("cuda")  # 采用GPU训练


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    """
    dataloader:加载定义的数据集
    """
    dataset_train = dataloader.train_loader
    dataset_test = dataloader.test_loader

    """
    输出训练集和测试的图像数量
    """
    date_train_size = len(dataset_train)
    date_test_size = len(dataset_test)
    print("训练集长度为：{}".format(date_train_size))
    print("测试集长度为：{}".format(date_test_size))

    """
    创建神经网络
    """
    # ziya = model_fsrcnn.Net()
    ziya = model_3dunet.UNet3DSR()

    ziya = ziya.to(device)

    """
    定义损失函数
    """
    loss_func = torch.nn.MSELoss()
    loss_func = loss_func.to(device)

    """
    定义psnr
    """


    def torchPSNR(tar_img, prd_img):
        imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
        rmse = (imdff ** 2).mean().sqrt()
        ps = 20 * torch.log10(1 / rmse)
        return ps


    """
    定义优化器
    """
    # fsrcnn
    # learn_rate = 1e-3
    # 3d-sr-unet
    learn_rate = 1e-3
    optimizer = torch.optim.Adam(params=ziya.parameters(), lr=learn_rate)

    """
    开始训练
    """
    psnr_test = 0
    total_step = 0

    torch.cuda.empty_cache()

    # tensorboard做可视化
    writer = SummaryWriter('./path/to/log')


    # 训练集上的表现
    epoch = 50  # 训练轮数
    for i in range(epoch):
        # loss = 0.0
        print("-------第{}轮训练开始--------".format(i + 1))
        for label in dataloader.train_loader:
            img_hr = label
            img_hr = img_hr.to(device)



            # 在网络内进行平均池化下采样，所以直接把高分辨图送进网络即可
            output = ziya(img_hr)

            loss = loss_func(output, img_hr)
            optimizer.zero_grad()
            loss.backward()  # 反向传播
            optimizer.step()  # 梯度下降
            total_step += 1
            writer.add_scalar("loss", loss, total_step)  # 将loss可视化

            if total_step % 32 ==0:
                print("--第{}次训练的loss为：{}--".format(total_step, loss.item()))


    # 测试过程
    with torch.no_grad():
        total_loss = 0
        total_psnr = 0

        count = 0

        for data in dataloader.test_loader:
            img_hr = data
            img_hr = img_hr.to(device)

            output = ziya(img_hr)


            loss = loss_func(output, img_hr)
            psnr = torchPSNR(img_hr, output)
            total_loss += loss
            total_psnr += psnr

            # output_cpu = output.cpu()
            output = output / 2 + 0.5

            output_cpu = output.cpu()
            output_np = output_cpu.numpy()

            # output_np_final = output_np * 255
            # output = np.transpose(output_np, (1,2,0))
            io.imsave('D:/fsrcnn_demo2/result/' + str(count) + '.tif', output_np)
            count += 1


            print("测试集平均损失为：{}".format(total_loss / len(dataloader.test_loader)))
            print("测试集平均psnr为：{}".format(total_psnr / len(dataloader.test_loader)))
            writer.add_scalar("psnr", total_psnr / len(dataloader.test_loader), count)  # 可视化
            writer.add_scalar("ave_loss", total_loss / len(dataloader.test_loader), count)  # 可视化

        if total_psnr / date_test_size > 0:
            torch.save(ziya, "ziya_3DSRUNET3_.pth")
            print("模型已保存")  # 若PSNR比上一次提高则保存模型
            psnr_test = total_psnr / date_test_size



        writer.close()